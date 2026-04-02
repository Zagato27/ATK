"""
ATK G-Eval — custom G-Eval implementation based on the original paper
(Liu et al., 2023 "G-EVAL: NLG Evaluation using GPT-4 with Better Human Alignment").

Key components from the paper:
1. Task Introduction + Evaluation Criteria — prompt defining the task
2. Auto Chain-of-Thoughts — LLM generates evaluation steps from criteria
3. Scoring Function — form-filling paradigm with multi-sample averaging

ATK G-Eval — собственная реализация G-Eval по оригинальной статье.
Три компонента: промпт с критериями, автогенерация CoT-шагов, form-filling
парадигма с multi-sample усреднением.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent_test_kit.judge import BaseLLMJudge

logger = logging.getLogger(__name__)

_COT_CACHE: dict[str, list[str]] = {}


@dataclass
class GEvalResult:
    """Result of a G-Eval evaluation. / Результат G-Eval оценки."""

    score: float
    raw_scores: list[int] = field(default_factory=list)
    passed: bool = False
    reasoning: str = ""
    evaluation_steps: list[str] = field(default_factory=list)
    raw_responses: list[str] = field(default_factory=list)


class ATKGEval:
    """Custom G-Eval implementation following the original paper.

    Собственная реализация G-Eval по оригинальной статье.
    """

    SYSTEM_PROMPT_V1 = (
        "Вы — экспертный оценщик качества ответов AI-агентов. "
        "Ваша задача — внимательно прочитать инструкции, критерии оценки "
        "и шаги оценивания, затем выставить оценку строго по шкале."
    )

    COT_GENERATION_PROMPT = (
        "Дано описание критерия оценки:\n"
        "{criteria}\n\n"
        "Сгенерируйте пошаговую инструкцию (3-5 шагов) для оценки ответа "
        "AI-агента по данному критерию. Каждый шаг должен быть конкретным "
        "и проверяемым.\n\n"
        "Формат ответа — ТОЛЬКО нумерованный список, без преамбулы:\n"
        "1. ...\n"
        "2. ...\n"
        "3. ..."
    )

    EVAL_PROMPT_TEMPLATE = (
        "Вам будет предоставлен один ответ AI-агента на сообщение пользователя.\n"
        "Ваша задача — оценить ответ по одной метрике.\n\n"
        "Пожалуйста, внимательно прочитайте и поймите эти инструкции.\n"
        "Держите этот документ открытым во время оценки и обращайтесь "
        "к нему по мере необходимости.\n\n"
        "Критерии оценки:\n"
        "{metric_name} (1-{scale}) — {criteria_text}\n\n"
        "Шаги оценивания:\n"
        "{evaluation_steps}\n\n"
        "{context_block}"
        "Сообщение пользователя:\n"
        "{input_text}\n\n"
        "Ответ агента:\n"
        "{output_text}\n\n"
        "Форма оценки (ТОЛЬКО оценки):\n"
        "- {metric_name}:"
    )

    def __init__(
        self,
        judge: BaseLLMJudge,
        *,
        n_samples: int = 1,
        score_scale: int = 5,
        temperature: float = 1.0,
        cot_cache_dir: str | None = None,
        system_prompt_version: str = "v1",
        require_reasoning: bool = True,
        reasoning_backfill_attempts: int = 2,
        verbose_logging: bool = False,
        log_prompt_chars: int = 4000,
        log_response_chars: int = 4000,
    ):
        self._judge = judge
        self._n_samples = n_samples
        self._score_scale = score_scale
        self._temperature = temperature
        self._cot_cache_dir = Path(cot_cache_dir) if cot_cache_dir else None
        self._system_prompt_version = system_prompt_version
        self._require_reasoning = require_reasoning
        self._reasoning_backfill_attempts = max(1, reasoning_backfill_attempts)
        self._verbose_logging = verbose_logging
        self._log_prompt_chars = log_prompt_chars
        self._log_response_chars = log_response_chars

    @property
    def system_prompt(self) -> str:
        prompts = {"v1": self.SYSTEM_PROMPT_V1}
        return prompts.get(self._system_prompt_version, self.SYSTEM_PROMPT_V1)

    def generate_evaluation_steps(self, criteria: str) -> list[str]:
        """Auto CoT: generate evaluation steps from criteria and cache them.

        Auto CoT: генерирует шаги оценки из критериев и кэширует результат.
        """
        cache_key = hashlib.sha256(criteria.encode()).hexdigest()[:16]

        if cache_key in _COT_CACHE:
            return _COT_CACHE[cache_key]

        if self._cot_cache_dir:
            cache_file = self._cot_cache_dir / f"{cache_key}.json"
            if cache_file.exists():
                try:
                    steps = json.loads(cache_file.read_text(encoding="utf-8"))
                    _COT_CACHE[cache_key] = steps
                    return steps
                except (json.JSONDecodeError, OSError):
                    pass

        prompt = self.COT_GENERATION_PROMPT.format(criteria=criteria)
        if self._should_log_verbose():
            logger.debug(
                "ATKGEval CoT generation criteria:\n%s",
                self._short(criteria, self._log_prompt_chars),
            )
            logger.debug(
                "ATKGEval CoT generation prompt:\n%s",
                self._short(prompt, self._log_prompt_chars),
            )
        raw = self._judge.generate(self._with_system_prompt(prompt))
        steps = self._parse_steps(raw)

        if not steps:
            steps = [
                "1. Внимательно прочитайте сообщение пользователя.",
                "2. Прочитайте ответ агента и сопоставьте с критерием.",
                f"3. Выставьте оценку от 1 до {self._score_scale}.",
            ]
            logger.warning("CoT generation returned no steps, using fallback")

        _COT_CACHE[cache_key] = steps

        if self._cot_cache_dir:
            try:
                self._cot_cache_dir.mkdir(parents=True, exist_ok=True)
                cache_file = self._cot_cache_dir / f"{cache_key}.json"
                cache_file.write_text(
                    json.dumps(steps, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except OSError as exc:
                logger.warning("Failed to write CoT cache: %s", exc)

        if self._should_log_verbose():
            logger.debug("ATKGEval evaluation steps:\n%s", "\n".join(steps))

        return steps

    def evaluate(
        self,
        input_text: str,
        output_text: str,
        criteria: str,
        threshold: float,
        *,
        metric_name: str = "quality",
        context: list[str] | None = None,
        expected_output: str | None = None,
        n_samples: int | None = None,
    ) -> GEvalResult:
        """Full G-Eval pipeline per the original paper.

        Полный G-Eval pipeline по оригинальной статье.
        """
        n = n_samples or self._n_samples
        steps = self.generate_evaluation_steps(criteria)
        prompt = self._build_prompt(
            criteria=criteria,
            steps=steps,
            input_text=input_text,
            output_text=output_text,
            context=context,
            expected_output=expected_output,
            metric_name=metric_name,
        )

        if self._should_log_verbose():
            logger.debug(
                "ATKGEval evaluate: metric=%s threshold=%.3f n_samples=%d",
                metric_name,
                threshold,
                n,
            )
            logger.debug(
                "ATKGEval criteria:\n%s",
                self._short(criteria, self._log_prompt_chars),
            )
            if context:
                logger.debug(
                    "ATKGEval context:\n%s",
                    self._short("\n".join(context), self._log_prompt_chars),
                )
            if expected_output:
                logger.debug(
                    "ATKGEval expected output:\n%s",
                    self._short(expected_output, self._log_response_chars),
                )
            logger.debug(
                "ATKGEval chat user:\n%s",
                self._short(input_text, self._log_prompt_chars),
            )
            logger.debug(
                "ATKGEval chat agent:\n%s",
                self._short(output_text, self._log_prompt_chars),
            )
            logger.debug(
                "ATKGEval full prompt:\n%s",
                self._short(prompt, self._log_prompt_chars),
            )

        raw_scores: list[int] = []
        raw_responses: list[str] = []
        reasoning = ""

        for i in range(n):
            try:
                response = self._judge.generate(self._with_system_prompt(prompt))
            except Exception as exc:
                logger.warning("G-Eval sample %d/%d failed: %s", i + 1, n, exc)
                raw_responses.append(f"ERROR: {exc}")
                continue

            raw_responses.append(response)
            score = self._parse_score(response, metric_name=metric_name)
            if self._should_log_verbose():
                logger.debug(
                    "ATKGEval raw judge response sample %d/%d:\n%s",
                    i + 1,
                    n,
                    self._short(response, self._log_response_chars),
                )
                logger.debug(
                    "ATKGEval parsed score sample %d/%d: %s",
                    i + 1,
                    n,
                    score,
                )
            if score is not None:
                raw_scores.append(score)
                if not reasoning:
                    reasoning = self._extract_reasoning(
                        response, metric_name=metric_name
                    )

        if not raw_scores:
            logger.error("All %d G-Eval samples failed to produce a score", n)
            return GEvalResult(
                score=0.0,
                raw_scores=[],
                passed=False,
                reasoning="",
                evaluation_steps=steps,
                raw_responses=raw_responses,
            )

        avg = sum(raw_scores) / len(raw_scores)
        normalized = (avg - 1) / max(self._score_scale - 1, 1)
        score_value = round(avg, 2)

        if self._require_reasoning and not reasoning:
            for attempt in range(1, self._reasoning_backfill_attempts + 1):
                reasoning_prompt = self._build_reasoning_prompt(
                    criteria=criteria,
                    input_text=input_text,
                    output_text=output_text,
                    metric_name=metric_name,
                    score_value=score_value,
                    attempt=attempt,
                )
                if self._should_log_verbose():
                    logger.debug(
                        "ATKGEval reasoning backfill prompt (attempt %d/%d):\n%s",
                        attempt,
                        self._reasoning_backfill_attempts,
                        self._short(reasoning_prompt, self._log_prompt_chars),
                    )
                try:
                    reasoning_response = self._judge.generate(
                        self._with_system_prompt(reasoning_prompt)
                    )
                    raw_responses.append(reasoning_response)
                    extracted = self._extract_reasoning(
                        reasoning_response, metric_name=metric_name
                    )
                    reasoning = extracted
                    if self._should_log_verbose():
                        logger.debug(
                            "ATKGEval reasoning backfill response (attempt %d/%d):\n%s",
                            attempt,
                            self._reasoning_backfill_attempts,
                            self._short(reasoning_response, self._log_response_chars),
                        )
                    if reasoning:
                        break
                except Exception as exc:
                    logger.warning("ATKGEval reasoning backfill failed: %s", exc)

        reasoning_ok = bool(reasoning)
        if not reasoning_ok:
            reasoning = "Обоснование не предоставлено моделью."
            if self._require_reasoning:
                logger.warning(
                    "ATKGEval: missing textual reasoning after %d backfill attempts",
                    self._reasoning_backfill_attempts,
                )
        final_passed = (normalized >= threshold) and (
            (not self._require_reasoning) or reasoning_ok
        )
        if self._should_log_verbose():
            logger.debug(
                "ATKGEval result: normalized_score=%.3f raw_score_avg=%.2f raw_scores=%s threshold=%.3f passed=%s",
                normalized,
                score_value,
                raw_scores,
                threshold,
                final_passed,
            )
            logger.debug(
                "ATKGEval reasoning:\n%s",
                self._short(reasoning, self._log_response_chars),
            )

        return GEvalResult(
            score=normalized,
            raw_scores=raw_scores,
            passed=final_passed,
            reasoning=reasoning,
            evaluation_steps=steps,
            raw_responses=raw_responses,
        )

    def _build_prompt(
        self,
        criteria: str,
        steps: list[str],
        input_text: str,
        output_text: str,
        context: list[str] | None,
        expected_output: str | None = None,
        metric_name: str = "quality",
    ) -> str:
        steps_text = "\n".join(steps)

        context_block = ""
        if context:
            context_block = (
                "Контекст:\n" + "\n".join(context) + "\n\n"
            )
        if expected_output:
            context_block += (
                "Ожидаемый ответ:\n" + expected_output + "\n\n"
            )

        return self.EVAL_PROMPT_TEMPLATE.format(
            metric_name=metric_name,
            scale=self._score_scale,
            criteria_text=criteria,
            evaluation_steps=steps_text,
            context_block=context_block,
            input_text=input_text,
            output_text=output_text,
        )

    def _build_reasoning_prompt(
        self,
        *,
        criteria: str,
        input_text: str,
        output_text: str,
        metric_name: str,
        score_value: float,
        attempt: int = 1,
    ) -> str:
        """Build a follow-up prompt to request textual justification."""
        strict_tail = (
            "Важно: НЕ возвращай строку с оценкой, НЕ повторяй '<metric>: <score>'. "
            "Верни только строку '- reasoning: ...'."
        )
        if attempt > 1:
            strict_tail = (
                "Это повторный запрос. В предыдущем ответе не было обоснования. "
                "Обязательно верни только обоснование в формате "
                "'- reasoning: <1-3 предложения>'. "
                "Запрещено возвращать числовую оценку."
            )
        return (
            "Ты уже оценил ответ агента по метрике.\n"
            "Теперь дай краткое и конкретное обоснование этой оценки (1-3 предложения).\n\n"
            f"Метрика: {metric_name}\n"
            f"Поставленная оценка: {score_value}\n\n"
            "Критерии оценки:\n"
            f"{criteria}\n\n"
            "Сообщение пользователя:\n"
            f"{input_text}\n\n"
            "Ответ агента:\n"
            f"{output_text}\n\n"
            "Формат ответа:\n"
            "- reasoning: <текст обоснования>\n\n"
            f"{strict_tail}"
        )

    def _extract_reasoning(
        self,
        response: str,
        *,
        metric_name: str | None = None,
    ) -> str:
        """Extract textual reasoning from judge response."""
        lines = [line.rstrip() for line in response.splitlines() if line.strip()]
        if not lines:
            return ""

        reason_pattern = re.compile(
            r"^\s*-\s*(reasoning|обоснование)\s*:\s*(.*)$",
            re.IGNORECASE,
        )
        collected: list[str] = []
        collecting = False
        for line in lines:
            m = reason_pattern.match(line)
            if m:
                collecting = True
                first = m.group(2).strip()
                if first:
                    collected.append(first)
                continue
            if collecting:
                # Continue multiline reasoning until another field marker.
                if re.match(r"^\s*-\s*[a-zA-Zа-яА-Я_]+\s*:", line):
                    break
                collected.append(line.strip())
        if collected:
            return "\n".join(collected).strip()

        score_line = None
        if metric_name:
            score_line = re.compile(
                rf"^\s*-\s*{re.escape(metric_name)}\s*:\s*\d+\s*[\.!\?]?\s*$",
                re.IGNORECASE,
            )

        filtered = []
        for line in lines:
            if score_line and score_line.match(line):
                continue
            filtered.append(line.strip())
        return "\n".join(filtered).strip()

    def _parse_score(
        self,
        response: str,
        metric_name: str | None = None,
    ) -> int | None:
        """Extract integer score (1-scale) from judge response.

        Извлекает целый скор из ответа судьи. Regex-fallback при невалидном формате.
        """
        if metric_name:
            strict_pattern = re.compile(
                rf"^\s*-\s*{re.escape(metric_name)}\s*:\s*(\d+)\s*[\.!\?]?\s*$",
                re.IGNORECASE,
            )
            for line in response.splitlines():
                match = strict_pattern.match(line.strip())
                if not match:
                    continue
                val = int(match.group(1))
                if 1 <= val <= self._score_scale:
                    return val

        for match in re.finditer(r"\b(\d+)\b", response):
            val = int(match.group(1))
            if 1 <= val <= self._score_scale:
                return val

        logger.warning("Could not parse score from response: %s", response[:200])
        return None

    def _with_system_prompt(self, prompt: str) -> str:
        """Inject a system-level instruction into single-string judge APIs."""
        return (
            f"[SYSTEM]\n{self.system_prompt}\n\n"
            f"[USER]\n{prompt}"
        )

    def _should_log_verbose(self) -> bool:
        """Enable verbose logs explicitly or when logger runs in DEBUG mode."""
        return self._verbose_logging or logger.isEnabledFor(logging.DEBUG)

    @staticmethod
    def _short(text: str, limit: int) -> str:
        """Return text truncated to *limit* chars with a marker."""
        if limit <= 0 or len(text) <= limit:
            return text
        return f"{text[:limit]}...<truncated {len(text) - limit} chars>"

    @staticmethod
    def _parse_steps(raw: str) -> list[str]:
        """Parse numbered list from LLM output. / Парсит нумерованный список из ответа LLM."""
        lines = raw.strip().splitlines()
        steps = []
        for line in lines:
            stripped = line.strip()
            if re.match(r"^\d+[\.\)]\s+", stripped):
                steps.append(stripped)
        return steps
