"""
Generic communication style tests applicable to any LLM agent.

Thresholds are read from ``agent-test-kit.toml`` → ``[style]``.

Requires a ``session`` fixture returning an initialized
:class:`~agent_test_kit.AgentSession` with an LLM judge configured.
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


class GenericStyleTests:
    """Communication style assertions for any LLM agent."""

    _GUIDANCE_CASES = {
        "I need help": "неясный запрос с просьбой о помощи",
        "Help": "очень короткий неясный запрос",
    }

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

        if "timed out" in lower or "timeout" in lower:
            return (
                "Превышен таймаут",
                "Проверка стиля общения не завершилась вовремя, поэтому результат "
                "нельзя интерпретировать надёжно.",
            )
        if self._looks_like_environment_error(message):
            return (
                "Проблема окружения или доступности сервиса",
                "Проверка прервалась из-за сетевой или серверной ошибки, а не "
                "из-за самой коммуникативной манеры агента.",
            )
        if "response text is empty" in lower:
            return (
                "Пустой ответ",
                "Агент формально ответил, но не вернул пользовательского текста, "
                "поэтому стиль ответа оценить нельзя.",
            )
        if lower.startswith("turn") and "http " in lower:
            return (
                "Сервис вернул ошибку",
                "Во время style-сценария сервис не отдал успешный HTTP-ответ.",
            )
        if "evaluate_direct(" in lower or ("score=" in lower and "threshold=" in lower):
            if case == "guidance":
                return (
                    "Оценка judge ниже порога",
                    "Judge-проверка показала, что ответ недостаточно помогает "
                    "пользователю с неясным запросом.",
                )
            return (
                "Оценка judge ниже порога",
                "Judge-проверка показала, что ответ недостаточно вежлив или "
                "стилистически корректен.",
            )
        return (
            "Нарушено ожидаемое поведение",
            "Результат проверки не соответствует ожидаемому стилю общения агента.",
        )

    def _dialogue_markdown(self, session) -> str:
        history = list(getattr(session, "history", []))
        init_message = getattr(session, "init_message", "")
        if (
            history
            and init_message
            and history[0].get("role") == "assistant"
            and history[0].get("content") == init_message
        ):
            history = history[1:]

        lines = ["# Короткий диалог", ""]
        user_turn = 0
        for item in history:
            role = item.get("role", "unknown")
            content = str(item.get("content", ""))
            if role == "user":
                user_turn += 1
                title = f"Пользователь, шаг {user_turn}"
            else:
                title = f"Агент, шаг {user_turn or 0}"
            lines.extend([f"## {title}", "", "```text", content, "```", ""])
        return "\n".join(lines).strip() + "\n"

    def _case_markdown(
        self,
        *,
        case_type: str,
        case_name: str,
        metric: str,
        threshold: float,
        prompt: str,
        session,
    ) -> str:
        last_text = self._safe_last_text(session)
        rows = [
            ("Тип кейса", case_type),
            ("Сценарий", case_name),
            ("Метрика judge", metric),
            ("Порог judge", f"{threshold:.2f}"),
            ("Пользовательский запрос", self._preview_text(prompt, limit=220) or "-"),
            ("Последний ответ агента", self._preview_text(last_text) or "-"),
            ("Длина ответа", str(len(last_text or ""))),
            ("Ответ заканчивается вопросом", "да" if (last_text or "").rstrip().endswith("?") else "нет"),
            ("Текущий ход", str(getattr(session, "turn", "-"))),
        ]

        result = getattr(session, "last_eval_result", None)
        if result is not None:
            rows.extend(
                [
                    ("Judge score", f"{getattr(result, 'score', 0.0):.3f}"),
                    ("Judge passed", "да" if getattr(result, "passed", False) else "нет"),
                ]
            )

        lines = ["# Детали style case", "", "| Поле | Значение |", "|---|---|"]
        for key, value in rows:
            lines.append(
                f"| {key} | {str(value).replace('|', '/').replace(chr(10), ' ')} |"
            )
        return "\n".join(lines) + "\n"

    def _build_actual(
        self,
        *,
        session,
        threshold: float,
        extra: list[str] | None = None,
    ) -> list[str]:
        actual = [
            f"Текущий номер хода: {getattr(session, 'turn', '-')}",
            f"Сообщений в диалоге: {len(getattr(session, 'history', []))}",
            f"Порог judge: {threshold:.2f}",
        ]
        last_text = self._safe_last_text(session)
        if last_text:
            actual.append(f"Последний ответ агента: {self._preview_text(last_text)}")
            actual.append(
                "Ответ заканчивается вопросом: "
                + ("да" if last_text.rstrip().endswith("?") else "нет")
            )

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
        verdict_title: str,
        verdict_text: str,
        actual: list[str],
        findings: list[str],
        case_type: str,
        case_name: str,
        metric: str,
        threshold: float,
        prompt: str,
        session,
        technical: dict[str, Any],
    ) -> None:
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
        attach_markdown("dialogue.md", self._dialogue_markdown(session))
        attach_markdown(
            "case.md",
            self._case_markdown(
                case_type=case_type,
                case_name=case_name,
                metric=metric,
                threshold=threshold,
                prompt=prompt,
                session=session,
            ),
        )
        if findings:
            attach_text("findings.txt", "\n".join(f"- {item}" for item in findings))
        attach_json("[tech] style-case.json", technical)

    @staticmethod
    def _eval_result_data(session) -> dict[str, Any] | None:
        result = getattr(session, "last_eval_result", None)
        if result is None:
            return None
        return {
            "score": getattr(result, "score", None),
            "passed": getattr(result, "passed", None),
            "reasoning": getattr(result, "reasoning", None),
            "raw_scores": getattr(result, "raw_scores", None),
        }

    def _run_style_case(
        self,
        *,
        session,
        prompt: str,
        metric: str,
        threshold: float,
        title: str,
        scenario: str,
        expected: list[str],
        case_type: str,
        case_name: str,
    ) -> None:
        try:
            with step("Отправка пользовательского запроса"):
                session.send(prompt)

            with step("Проверка базовой корректности ответа"):
                session.expect_response_ok()

            with step("Judge оценивает стиль ответа"):
                session.evaluate(metric, threshold=threshold)
        except Exception as exc:
            verdict_title, verdict_text = self._classify_failure(
                exc,
                case="guidance" if metric == "guidance" else "politeness",
            )
            self._attach_case_report(
                title=title,
                scenario=scenario,
                expected=expected,
                verdict_title=verdict_title,
                verdict_text=verdict_text,
                actual=self._build_actual(
                    session=session,
                    threshold=threshold,
                    extra=[
                        f"Ошибка: {self._preview_text(str(exc), limit=220)}",
                    ],
                ),
                findings=[
                    "Judge или базовая проверка не подтвердили ожидаемый стиль коммуникации.",
                ],
                case_type=case_type,
                case_name=case_name,
                metric=metric,
                threshold=threshold,
                prompt=prompt,
                session=session,
                technical={
                    "case": "style",
                    "case_type": case_type,
                    "case_name": case_name,
                    "metric": metric,
                    "threshold": threshold,
                    "prompt": prompt,
                    "response_preview": self._preview_text(self._safe_last_text(session), limit=300),
                    "response_length": len(self._safe_last_text(session) or ""),
                    "turn": getattr(session, "turn", None),
                    "eval_result": self._eval_result_data(session),
                },
            )
            raise

        self._attach_case_report(
            title=title,
            scenario=scenario,
            expected=expected,
            verdict_title="Проверка пройдена",
            verdict_text="Ответ соответствует ожидаемому стилю общения для данного сценария.",
            actual=self._build_actual(
                session=session,
                threshold=threshold,
            ),
            findings=[
                "Коммуникативная манера агента соответствует ожидаемому сценарию.",
            ],
            case_type=case_type,
            case_name=case_name,
            metric=metric,
            threshold=threshold,
            prompt=prompt,
            session=session,
            technical={
                "case": "style",
                "case_type": case_type,
                "case_name": case_name,
                "metric": metric,
                "threshold": threshold,
                "prompt": prompt,
                "response_preview": self._preview_text(self._safe_last_text(session), limit=300),
                "response_length": len(self._safe_last_text(session) or ""),
                "turn": getattr(session, "turn", None),
                "eval_result": self._eval_result_data(session),
            },
        )

    @pytest.mark.judge
    def test_greeting_politeness(self, session):
        """Agent responds to greeting politely. / Агент вежливо отвечает на приветствие."""
        cfg = get_config().style
        self._run_style_case(
            session=session,
            prompt="Hello!",
            metric="politeness",
            threshold=cfg.politeness_threshold,
            title="Агент вежливо отвечает на приветствие",
            scenario=(
                "Пользователь пишет обычное приветствие. Ответ должен быть "
                "вежливым и корректным по тону."
            ),
            expected=[
                "Агент возвращает корректный непустой ответ",
                f"Judge подтверждает вежливый тон с порогом не ниже {cfg.politeness_threshold:.2f}",
            ],
            case_type="Вежливость",
            case_name="ответ на приветствие",
        )

    @pytest.mark.judge
    def test_polite_under_rude_input(self, session):
        """Agent stays polite even when the user is rude. / Агент остаётся вежливым даже при грубом вводе пользователя."""
        cfg = get_config().style
        self._run_style_case(
            session=session,
            prompt="This is terrible, you are useless, just do your job!",
            metric="politeness",
            threshold=cfg.politeness_threshold,
            title="Агент сохраняет вежливость при грубом сообщении",
            scenario=(
                "Пользователь пишет грубое и токсичное сообщение. Агент не должен "
                "зеркалить агрессию и должен сохранить вежливый тон."
            ),
            expected=[
                "Агент возвращает корректный непустой ответ",
                f"Judge подтверждает вежливый тон с порогом не ниже {cfg.politeness_threshold:.2f}",
            ],
            case_type="Вежливость",
            case_name="вежливость при грубом вводе",
        )

    @pytest.mark.judge
    @pytest.mark.parametrize(
        "prompt",
        [
            pytest.param("I need help", id="vague_help"),
            pytest.param("Help", id="short_help"),
        ],
    )
    def test_vague_request_gets_guidance(self, session, prompt):
        """Vague requests receive actionable guidance. / Неясный запрос получает направляющий и полезный ответ."""
        cfg = get_config().style
        case_name = self._GUIDANCE_CASES.get(prompt, "неясный запрос")
        self._run_style_case(
            session=session,
            prompt=prompt,
            metric="guidance",
            threshold=cfg.guidance_threshold,
            title=f"Агент даёт понятное направление: {case_name}",
            scenario=(
                "Пользователь формулирует слишком общий запрос. Ответ должен не "
                "просто подтверждать запрос, а помогать сделать следующий шаг."
            ),
            expected=[
                "Агент возвращает корректный непустой ответ",
                f"Judge подтверждает полезный направляющий ответ с порогом не ниже {cfg.guidance_threshold:.2f}",
            ],
            case_type="Guidance",
            case_name=case_name,
        )
