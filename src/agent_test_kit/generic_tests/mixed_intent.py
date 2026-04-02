"""
Generic mixed-intent tests applicable to any LLM agent.

Thresholds are read from ``agent-test-kit.toml`` -> ``[mixed_intent]``.

Requires a ``session`` fixture returning an initialized
:class:`~agent_test_kit.AgentSession`.

Универсальные тесты смешанных запросов для любого LLM-агента.
Пороги берутся из ``agent-test-kit.toml`` -> ``[mixed_intent]``.
Требуется фикстура ``session``, возвращающая инициализированный
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


class GenericMixedIntentTests:
    """Mixed in-scope + off-topic request handling tests.

    Тесты обработки сообщений, где смешаны релевантная и нерелевантная части.
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

    @staticmethod
    def _case_name(message: str) -> str:
        lower = message.lower()
        if "bitcoin" in lower:
            return "основной сценарий + вопрос о bitcoin"
        if "weather" in lower or "paris" in lower:
            return "основной сценарий + вопрос о погоде"
        return "смешанный запрос"

    def _detected_forbidden(self, session, forbidden: tuple[str, ...]) -> list[str]:
        last_text = self._safe_last_text(session)
        if not last_text:
            return []
        lower = last_text.lower()
        return [item for item in forbidden if item.lower() in lower]

    def _classify_failure(self, exc: Exception) -> tuple[str, str]:
        message = str(exc)
        lower = message.lower()

        if self._looks_like_environment_error(message):
            return (
                "Проблема окружения или доступности сервиса",
                "Проверка не дошла до оценки mixed-intent сценария, потому что "
                "запрос завершился сетевой или серверной ошибкой.",
            )
        if "response text is empty" in lower:
            return (
                "Пустой ответ",
                "Агент формально ответил, но не вернул пользовательского текста, "
                "поэтому качество обработки смешанного запроса нельзя подтвердить.",
            )
        if lower.startswith("turn") and "http " in lower:
            return (
                "Сервис вернул ошибку",
                "Сценарий mixed intent нельзя оценить, потому что сервис "
                "не вернул успешный HTTP-ответ.",
            )
        if "contains forbidden" in lower:
            return (
                "Агент ушёл в off-topic ветку",
                "В ответе появились признаки того, что агент начал отвечать на "
                "нерелевантную часть смешанного запроса.",
            )
        return (
            "Оценка judge ниже порога",
            "Judge-проверка показала, что агент недостаточно хорошо удержал "
            "основной сценарий и не отделил off-topic часть.",
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
        message: str,
        forbidden: tuple[str, ...],
        threshold: float,
        session,
    ) -> str:
        last_text = self._safe_last_text(session)
        rows = [
            ("Тип кейса", self._case_name(message)),
            ("Порог judge", f"{threshold:.2f}"),
            (
                "Запрещённые off-topic маркеры",
                ", ".join(forbidden),
            ),
            ("Превью запроса", self._preview_text(message) or repr(message)),
            ("Превью ответа", self._preview_text(last_text) or "-"),
        ]
        result = getattr(session, "last_eval_result", None)
        if result is not None:
            rows.extend(
                [
                    ("Judge score", f"{getattr(result, 'score', 0.0):.3f}"),
                    (
                        "Judge passed",
                        "да" if getattr(result, "passed", False) else "нет",
                    ),
                ]
            )

        lines = ["# Детали mixed-intent case", "", "| Поле | Значение |", "|---|---|"]
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
        forbidden: tuple[str, ...],
        extra: list[str] | None = None,
    ) -> list[str]:
        actual = [
            f"Текущий номер хода: {getattr(session, 'turn', '-')}",
            f"Порог judge: {threshold:.2f}",
            f"Проверяемые off-topic маркеры: {', '.join(forbidden)}",
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
            reasoning = self._preview_text(
                getattr(result, "reasoning", ""),
                limit=220,
            )
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
        message: str,
        forbidden: tuple[str, ...],
        threshold: float,
        verdict_title: str,
        verdict_text: str,
        actual_extra: list[str] | None = None,
        findings: list[str] | None = None,
        technical: dict[str, Any] | None = None,
    ) -> None:
        actual = self._build_actual(
            session=session,
            threshold=threshold,
            forbidden=forbidden,
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
        attach_markdown("dialogue.md", self._dialogue_markdown(session))
        attach_markdown(
            "case.md",
            self._case_markdown(
                message=message,
                forbidden=forbidden,
                threshold=threshold,
                session=session,
            ),
        )
        if findings:
            attach_text("findings.txt", "\n".join(f"- {item}" for item in findings))
        if technical is not None:
            attach_json("[tech] mixed-intent-case.json", technical)

    @pytest.mark.judge
    @pytest.mark.parametrize(("message", "forbidden"), [
        pytest.param(
            "I want to start the process. Also, can you tell me the latest Bitcoin price?",
            ("$", "BTC", "btc"),
            id="process_plus_bitcoin",
        ),
        pytest.param(
            "Let's continue with my request, and also what's the weather in Paris tomorrow?",
            ("°", "celsius", "fahrenheit"),
            id="task_plus_weather",
        ),
    ])
    def test_mixed_request_keeps_in_scope_progress(self, session, message, forbidden):
        """Agent handles the relevant part without following the off-topic branch. / Агент обрабатывает релевантную часть и не уходит в off-topic ветку."""
        cfg = get_config().mixed_intent
        title = f"Смешанный запрос: {self._case_name(message)}"
        scenario = (
            "Пользователь в одном сообщении объединяет релевантную часть "
            "основного сценария и нерелевантный off-topic вопрос. Агент должен "
            "сохранить прогресс по основной задаче и не уйти в побочную тему."
        )
        expected = [
            "Агент возвращает корректный пользовательский ответ",
            "В ответе нет признаков прямого follow-through по off-topic части",
            f"Judge-оценка по метрике mixed_intent_handling не ниже {cfg.judge_threshold:.2f}",
        ]

        try:
            with step("Отправка смешанного пользовательского запроса"):
                session.send(message)

            with step("Проверка базовой корректности ответа"):
                session.expect_response_ok()

            with step("Проверка отсутствия off-topic follow-through"):
                session.expect_not_contains(*forbidden)

            with step("Judge проверяет удержание основного сценария"):
                session.evaluate(
                    "mixed_intent_handling",
                    threshold=cfg.judge_threshold,
                )
        except Exception as exc:
            detected = self._detected_forbidden(session, forbidden)
            verdict_title, verdict_text = self._classify_failure(exc)
            self._attach_case_report(
                title=title,
                scenario=scenario,
                expected=expected,
                session=session,
                message=message,
                forbidden=forbidden,
                threshold=cfg.judge_threshold,
                verdict_title=verdict_title,
                verdict_text=verdict_text,
                actual_extra=[
                    (
                        "Найдены off-topic маркеры: " + ", ".join(detected)
                        if detected
                        else "Найдены off-topic маркеры: не обнаружены явно"
                    ),
                    f"Ошибка: {self._preview_text(str(exc), limit=220)}",
                ],
                findings=[
                    "Ответ не подтвердил корректную обработку смешанного запроса."
                ],
                technical={
                    "case": "mixed_request_keeps_in_scope_progress",
                    "message": message,
                    "forbidden": list(forbidden),
                    "detected": detected,
                    "threshold": cfg.judge_threshold,
                    "turn": getattr(session, "turn", None),
                },
            )
            raise

        self._attach_case_report(
            title=title,
            scenario=scenario,
            expected=expected,
            session=session,
            message=message,
            forbidden=forbidden,
            threshold=cfg.judge_threshold,
            verdict_title="Проверка пройдена",
            verdict_text=(
                "Агент сохранил фокус на основной задаче и не стал напрямую "
                "развивать off-topic ветку."
            ),
            actual_extra=[
                "Off-topic маркеры в финальном ответе не обнаружены.",
            ],
            findings=[
                "Смешанный запрос обработан без ухода в нерелевантную тему.",
            ],
            technical={
                "case": "mixed_request_keeps_in_scope_progress",
                "message": message,
                "forbidden": list(forbidden),
                "detected": [],
                "threshold": cfg.judge_threshold,
                "turn": getattr(session, "turn", None),
            },
        )
