"""
Generic long-context tests applicable to any LLM agent.

Thresholds are read from ``agent-test-kit.toml`` → ``[long_context]``.

Requires a ``session`` fixture returning an initialized
:class:`~agent_test_kit.AgentSession`.
"""
from __future__ import annotations

from typing import Any

import pytest

from agent_test_kit.allure_support import (
    attach_json,
    attach_markdown,
    attach_text,
    set_description,
    set_title,
    step,
)
from agent_test_kit.config import get_config
from agent_test_kit.session import run_dialogue


def _advance_dialogue(session, turns: int) -> None:
    """Advance dialogue with lightweight distractor turns."""
    distractors = [
        "Let's continue",
        "Tell me more",
        "What should we do next?",
        "Can you clarify that?",
        "Please continue",
    ]
    messages = [distractors[i % len(distractors)] for i in range(turns)]
    run_dialogue(session, messages)


class GenericLongContextTests:
    """Long-context retention tests.

    Тесты удержания длинного контекста.
    """

    @staticmethod
    def _preview_text(text: str | None, limit: int = 180) -> str | None:
        if text is None:
            return None
        compact = " ".join(text.split())
        if not compact:
            compact = repr(text)
        if len(compact) <= limit:
            return compact
        return compact[: limit - 3] + "..."

    def _safe_last_text(self, session) -> str | None:
        try:
            return session.last_text
        except AssertionError:
            return None

    @staticmethod
    def _looks_like_environment_error(message: str) -> bool:
        lower = message.lower()
        return any(
            token in lower
            for token in (
                "502",
                "503",
                "504",
                "bad gateway",
                "proxyerror",
                "httpconnectionpool",
                "max retries exceeded",
                "connection reset",
                "connection refused",
                "timed out",
                "service unavailable",
                "gateway timeout",
            )
        )

    def _classify_failure(self, exc: Exception, *, case: str) -> tuple[str, str]:
        message = str(exc)
        lower = message.lower()

        if self._looks_like_environment_error(message):
            return (
                "Проблема окружения или доступности сервиса",
                "Проверка не дошла до оценки удержания длинного контекста, потому "
                "что запрос завершился сетевой или серверной ошибкой.",
            )
        if "response text is empty" in lower:
            return (
                "Пустой ответ",
                "Агент формально ответил, но не вернул пользовательского текста, "
                "поэтому удержание фактов подтвердить нельзя.",
            )
        if lower.startswith("turn") and "http " in lower:
            return (
                "Сервис вернул ошибку",
                "Сценарий длинного контекста нельзя оценить, потому что сервис "
                "не вернул успешный HTTP-ответ.",
            )
        if case in {"reference", "code"} and "missing keywords" in lower:
            return (
                "Факт потерялся в длинном контексте",
                "После серии отвлекающих ходов агент не смог воспроизвести "
                "ранее сохранённое значение.",
            )
        return (
            "Оценка judge ниже порога",
            "Семантическая проверка показала, что агент недостаточно хорошо "
            "удержал несколько фактов после длинного диалога.",
        )

    def _dialogue_digest_markdown(
        self,
        *,
        session,
        facts: list[str],
        distractor_turns: int,
    ) -> str:
        history = list(getattr(session, "history", []))
        init_message = getattr(session, "init_message", "")
        if (
            history
            and init_message
            and history[0].get("role") == "assistant"
            and history[0].get("content") == init_message
        ):
            history = history[1:]

        first_chunk = history[:6]
        last_chunk = history[-4:] if len(history) > 10 else history[6:]

        lines = [
            "# Выжимка длинного диалога",
            "",
            "## Ключевые факты",
            "",
        ]
        lines.extend(f"- {fact}" for fact in facts)
        lines.extend(
            [
                "",
                f"- Количество отвлекающих ходов: {distractor_turns}",
                "",
                "## Начало диалога",
                "",
            ]
        )
        for item in first_chunk:
            role = "Пользователь" if item.get("role") == "user" else "Агент"
            lines.extend([f"### {role}", "", "```text", str(item.get("content", "")), "```", ""])

        omitted = max(len(history) - len(first_chunk) - len(last_chunk), 0)
        if omitted:
            lines.extend(
                [
                    f"_Пропущено промежуточных сообщений: {omitted}_",
                    "",
                ]
            )

        if last_chunk:
            lines.extend(["## Конец диалога", ""])
            for item in last_chunk:
                role = "Пользователь" if item.get("role") == "user" else "Агент"
                lines.extend([f"### {role}", "", "```text", str(item.get("content", "")), "```", ""])

        return "\n".join(lines).strip() + "\n"

    def _case_markdown(
        self,
        *,
        memory_type: str,
        distractor_turns: int,
        facts: list[str],
        session,
        threshold: float | None = None,
    ) -> str:
        last_text = self._safe_last_text(session)
        rows = [
            ("Тип удерживаемой информации", memory_type),
            ("Количество отвлекающих ходов", str(distractor_turns)),
            ("Количество сообщений в истории", str(len(getattr(session, "history", [])))),
            ("Последний ответ агента", self._preview_text(last_text) or "-"),
        ]
        if threshold is not None:
            rows.append(("Порог judge", f"{threshold:.2f}"))
        result = getattr(session, "last_eval_result", None)
        if result is not None:
            rows.extend(
                [
                    ("Judge score", f"{getattr(result, 'score', 0.0):.3f}"),
                    ("Judge passed", "да" if getattr(result, "passed", False) else "нет"),
                ]
            )

        lines = ["# Детали long-context case", "", "| Поле | Значение |", "|---|---|"]
        for key, value in rows:
            lines.append(
                f"| {key} | {str(value).replace('|', '/').replace(chr(10), ' ')} |"
            )
        lines.extend(["", "## Ожидаемые факты", ""])
        lines.extend(f"- {fact}" for fact in facts)
        return "\n".join(lines) + "\n"

    def _build_actual(
        self,
        *,
        session,
        distractor_turns: int,
        extra: list[str] | None = None,
    ) -> list[str]:
        actual = [
            f"Текущий номер хода: {getattr(session, 'turn', '-')}",
            f"Количество сообщений в истории: {len(getattr(session, 'history', []))}",
            f"Количество отвлекающих ходов: {distractor_turns}",
        ]
        last_text = self._safe_last_text(session)
        if last_text:
            actual.append(f"Последний ответ агента: {self._preview_text(last_text)}")

        result = getattr(session, "last_eval_result", None)
        if result is not None:
            actual.append(f"Judge score: {getattr(result, 'score', 0.0):.3f}")
            actual.append(
                f"Judge passed: {'да' if getattr(result, 'passed', False) else 'нет'}"
            )
            reasoning = self._preview_text(getattr(result, "reasoning", ""), limit=220)
            if reasoning:
                actual.append(f"Judge reasoning: {reasoning}")

        if extra:
            actual.extend(extra)
        return actual

    def _attach_case_report(
        self,
        *,
        title: str,
        scenario: str,
        expected: list[str],
        session,
        facts: list[str],
        distractor_turns: int,
        memory_type: str,
        verdict_title: str,
        verdict_text: str,
        actual_extra: list[str] | None = None,
        findings: list[str] | None = None,
        technical: dict[str, Any] | None = None,
        threshold: float | None = None,
    ) -> None:
        actual = self._build_actual(
            session=session,
            distractor_turns=distractor_turns,
            extra=actual_extra,
        )
        summary_lines = [
            f"Сценарий:\n{scenario}",
            "",
            "Ожидалось:",
        ]
        summary_lines.extend(f"- {item}" for item in expected)
        summary_lines.extend(["", "Фактически:"])
        summary_lines.extend(f"- {item}" for item in actual)
        summary_lines.extend(["", f"Вывод:\n{verdict_title}. {verdict_text}"])
        summary = "\n".join(summary_lines)

        set_title(title)
        set_description(summary)
        attach_text("summary.txt", summary)
        attach_markdown(
            "dialogue.md",
            self._dialogue_digest_markdown(
                session=session,
                facts=facts,
                distractor_turns=distractor_turns,
            ),
        )
        attach_markdown(
            "case.md",
            self._case_markdown(
                memory_type=memory_type,
                distractor_turns=distractor_turns,
                facts=facts,
                session=session,
                threshold=threshold,
            ),
        )
        if findings:
            attach_text("findings.txt", "\n".join(f"- {item}" for item in findings))
        if technical is not None:
            attach_json("[tech] long-context-case.json", technical)

    def _run_recall_case(
        self,
        *,
        session,
        initial_prompt: str,
        final_prompt: str,
        expected_fact: str,
        distractor_turns: int,
        title: str,
        scenario: str,
        memory_type: str,
    ) -> None:
        facts = [expected_fact]
        try:
            with step("Сохранение исходного факта в диалоге"):
                session.send(initial_prompt)
                session.expect_response_ok()

            with step("Прогон отвлекающих ходов"):
                _advance_dialogue(session, distractor_turns)

            with step("Запрос на восстановление сохранённого факта"):
                session.send(final_prompt)
                session.expect_response_ok()

            with step("Проверка, что факт сохранился"):
                session.expect_contains(expected_fact)
        except Exception as exc:
            verdict_title, verdict_text = self._classify_failure(
                exc,
                case=memory_type,
            )
            self._attach_case_report(
                title=title,
                scenario=scenario,
                expected=[
                    "Агент корректно отвечает после исходного факта",
                    f"После {distractor_turns} отвлекающих ходов агент воспроизводит сохранённый факт",
                ],
                session=session,
                facts=facts,
                distractor_turns=distractor_turns,
                memory_type=memory_type,
                verdict_title=verdict_title,
                verdict_text=verdict_text,
                actual_extra=[
                    f"Исходный факт: {expected_fact}",
                    f"Финальный запрос: {final_prompt}",
                    f"Ошибка: {self._preview_text(str(exc), limit=220)}",
                ],
                findings=[
                    "После длинного контекста агент не подтвердил ранее сохранённый факт.",
                ],
                technical={
                    "case": memory_type,
                    "initial_prompt": initial_prompt,
                    "final_prompt": final_prompt,
                    "expected_fact": expected_fact,
                    "distractor_turns": distractor_turns,
                    "turn": getattr(session, "turn", None),
                },
            )
            raise

        self._attach_case_report(
            title=title,
            scenario=scenario,
            expected=[
                "Агент корректно отвечает после исходного факта",
                f"После {distractor_turns} отвлекающих ходов агент воспроизводит сохранённый факт",
            ],
            session=session,
            facts=facts,
            distractor_turns=distractor_turns,
            memory_type=memory_type,
            verdict_title="Проверка пройдена",
            verdict_text=(
                "Даже после длинного диалога агент сохранил и корректно "
                "воспроизвёл нужный факт."
            ),
            actual_extra=[
                f"Исходный факт: {expected_fact}",
                f"Финальный запрос: {final_prompt}",
            ],
            findings=[
                "Нужный факт сохранился после длинного контекста.",
            ],
            technical={
                "case": memory_type,
                "initial_prompt": initial_prompt,
                "final_prompt": final_prompt,
                "expected_fact": expected_fact,
                "distractor_turns": distractor_turns,
                "turn": getattr(session, "turn", None),
            },
        )

    def _run_semantic_case(
        self,
        *,
        session,
        fact_prompts: list[str],
        final_prompt: str,
        distractor_turns: int,
        threshold: float,
        title: str,
        scenario: str,
        memory_type: str,
        facts: list[str],
    ) -> None:
        try:
            for idx, prompt in enumerate(fact_prompts, start=1):
                with step(f"Сохранение факта {idx}"):
                    session.send(prompt)
                    session.expect_response_ok()

            with step("Прогон отвлекающих ходов"):
                _advance_dialogue(session, distractor_turns)

            with step("Финальный запрос на суммарное воспроизведение фактов"):
                session.send(final_prompt)
                session.expect_response_ok()

            with step("Judge проверяет удержание нескольких фактов"):
                session.evaluate("context_retention", threshold=threshold)
        except Exception as exc:
            verdict_title, verdict_text = self._classify_failure(exc, case=memory_type)
            self._attach_case_report(
                title=title,
                scenario=scenario,
                expected=[
                    "Агент корректно отвечает на каждый факт в начале диалога",
                    f"После {distractor_turns} отвлекающих ходов агент удерживает несколько фактов",
                    f"Judge-оценка по метрике context_retention не ниже {threshold:.2f}",
                ],
                session=session,
                facts=facts,
                distractor_turns=distractor_turns,
                memory_type=memory_type,
                verdict_title=verdict_title,
                verdict_text=verdict_text,
                actual_extra=[
                    f"Финальный запрос: {final_prompt}",
                    f"Порог judge: {threshold:.2f}",
                    f"Ошибка: {self._preview_text(str(exc), limit=220)}",
                ],
                findings=[
                    "Judge или базовая проверка не подтвердили сохранение нескольких фактов после длинного диалога.",
                ],
                technical={
                    "case": memory_type,
                    "fact_prompts": fact_prompts,
                    "final_prompt": final_prompt,
                    "facts": facts,
                    "distractor_turns": distractor_turns,
                    "threshold": threshold,
                    "turn": getattr(session, "turn", None),
                },
                threshold=threshold,
            )
            raise

        self._attach_case_report(
            title=title,
            scenario=scenario,
            expected=[
                "Агент корректно отвечает на каждый факт в начале диалога",
                f"После {distractor_turns} отвлекающих ходов агент удерживает несколько фактов",
                f"Judge-оценка по метрике context_retention не ниже {threshold:.2f}",
            ],
            session=session,
            facts=facts,
            distractor_turns=distractor_turns,
            memory_type=memory_type,
            verdict_title="Проверка пройдена",
            verdict_text=(
                "После длинного диалога агент сохранил несколько пользовательских "
                "фактов и успешно прошёл семантическую проверку."
            ),
            actual_extra=[
                f"Финальный запрос: {final_prompt}",
                f"Порог judge: {threshold:.2f}",
            ],
            findings=[
                "Несколько фактов сохранились после длинного контекста.",
            ],
            technical={
                "case": memory_type,
                "fact_prompts": fact_prompts,
                "final_prompt": final_prompt,
                "facts": facts,
                "distractor_turns": distractor_turns,
                "threshold": threshold,
                "turn": getattr(session, "turn", None),
            },
            threshold=threshold,
        )

    @pytest.mark.slow
    def test_reference_number_survives_10_turns(self, session):
        """Reference number survives 10 turns of dialogue. / Референсный номер сохраняется через 10 ходов диалога."""
        self._run_recall_case(
            session=session,
            initial_prompt="Important: my reference number is REF-98765",
            final_prompt="What was my reference number?",
            expected_fact="REF-98765",
            distractor_turns=9,
            title="Референсный номер сохраняется через 10 ходов",
            scenario=(
                "Пользователь сообщает референсный номер, затем происходит "
                "серия отвлекающих реплик, после чего агент должен вспомнить "
                "точное значение номера."
            ),
            memory_type="reference",
        )

    @pytest.mark.slow
    def test_code_survives_20_turns(self, session):
        """Code survives 20 turns of dialogue. / Код сохраняется через 20 ходов диалога."""
        self._run_recall_case(
            session=session,
            initial_prompt="Remember this code: ALPHA-2025",
            final_prompt="What code did I ask you to remember?",
            expected_fact="ALPHA-2025",
            distractor_turns=18,
            title="Код сохраняется через 20 ходов",
            scenario=(
                "Пользователь просит запомнить код, затем следует длинный "
                "диалог с отвлекающими репликами, после чего агент должен "
                "вспомнить точное значение кода."
            ),
            memory_type="code",
        )

    @pytest.mark.judge
    @pytest.mark.slow
    def test_multiple_facts_survive_long_dialogue_semantically(self, session):
        """Multiple facts remain available after a long dialogue. / Несколько фактов сохраняются после длинного диалога."""
        cfg = get_config().long_context
        self._run_semantic_case(
            session=session,
            fact_prompts=[
                "My name is Maria",
                "I live in Berlin",
                "I work as an engineer",
            ],
            final_prompt="Can you summarize what you know about me?",
            distractor_turns=8,
            threshold=cfg.judge_threshold,
            title="Несколько фактов сохраняются после длинного диалога",
            scenario=(
                "Пользователь сообщает несколько фактов о себе, затем следует "
                "длинный отвлекающий диалог, после чего агент должен "
                "семантически корректно воспроизвести накопленную информацию."
            ),
            memory_type="multiple-facts",
            facts=[
                "Имя: Maria",
                "Город: Berlin",
                "Профессия: engineer",
            ],
        )
