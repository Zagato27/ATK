"""
Generic session-resilience tests applicable to any LLM agent.

Requires a ``session`` fixture returning an initialized
:class:`~agent_test_kit.AgentSession`.

Универсальные тесты живучести сессии для любого LLM-агента.
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
from agent_test_kit.session import run_dialogue


class GenericSessionResilienceTests:
    """Session-liveness tests across multiple turns.

    Тесты живучести и работоспособности сессии после нескольких ходов.
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
                "service unavailable",
                "gateway timeout",
            )
        )

    def _classify_failure(self, exc: Exception) -> tuple[str, str]:
        message = str(exc)
        lower = message.lower()

        if "timed out" in lower or "timeout" in lower:
            return (
                "Превышен таймаут",
                "Проверка живучести сессии не завершилась вовремя, поэтому "
                "результат нельзя интерпретировать надёжно.",
            )
        if "session_id is not set" in lower:
            return (
                "Сессия не инициализирована или потеряна",
                "Проверочный probe не смог выполниться, потому что у клиента нет "
                "активного session_id.",
            )
        if self._looks_like_environment_error(message):
            return (
                "Проблема окружения или доступности сервиса",
                "Проверка живучести сессии прервалась из-за сетевой или серверной ошибки.",
            )
        if "response text is empty" in lower:
            return (
                "Пустой ответ",
                "На одном из ходов сервис формально ответил, но не вернул "
                "пользовательского текста, поэтому живучесть сессии подтвердить нельзя.",
            )
        if lower.startswith("turn") and "http " in lower:
            return (
                "Сервис вернул ошибку",
                "Во время многошагового сценария сервис не отдал успешный HTTP-ответ.",
            )
        if "no response from agent" in lower:
            return (
                "Сессия перестала отвечать",
                "На одном из этапов агент перестал возвращать ответ, поэтому "
                "сессию нельзя считать живой.",
            )
        return (
            "Нарушено ожидаемое поведение",
            "После серии сообщений сессия не подтвердила, что остаётся работоспособной.",
        )

    def _normalized_history(self, session) -> list[dict[str, Any]]:
        history = list(getattr(session, "history", []))
        init_message = getattr(session, "init_message", "")
        if (
            history
            and init_message
            and history[0].get("role") == "assistant"
            and history[0].get("content") == init_message
        ):
            history = history[1:]
        return history

    def _turns_markdown(self, session, *, planned_messages: list[str]) -> str:
        history = self._normalized_history(session)
        rows: list[tuple[str, str, str]] = []

        user_messages = [item for item in history if item.get("role") == "user"]
        assistant_messages = [item for item in history if item.get("role") == "assistant"]

        for idx, prompt in enumerate(planned_messages, start=1):
            reply = (
                str(assistant_messages[idx - 1].get("content", ""))
                if len(assistant_messages) >= idx
                else ""
            )
            rows.append(
                (
                    str(idx),
                    self._preview_text(prompt) or "-",
                    self._preview_text(reply) or "-",
                )
            )

        probe_sent = len(user_messages) > len(planned_messages)
        if probe_sent:
            probe_reply = (
                str(assistant_messages[len(planned_messages)].get("content", ""))
                if len(assistant_messages) > len(planned_messages)
                else ""
            )
            rows.append(
                (
                    "probe",
                    self._preview_text(str(user_messages[-1].get("content", ""))) or "test",
                    self._preview_text(probe_reply) or "-",
                )
            )

        lines = [
            "# Ходы сценария",
            "",
            "| Шаг | Сообщение пользователя | Превью ответа агента |",
            "|---|---|---|",
        ]
        for step_no, prompt, reply in rows:
            lines.append(
                f"| {step_no} | {prompt.replace('|', '/')} | {reply.replace('|', '/')} |"
            )
        return "\n".join(lines) + "\n"

    def _case_markdown(
        self,
        *,
        case_name: str,
        planned_messages: list[str],
        session,
    ) -> str:
        rows = [
            ("Сценарий", case_name),
            ("Плановых пользовательских сообщений", str(len(planned_messages))),
            (
                "Probe на живость выполнен",
                "да" if getattr(session, "turn", 0) > len(planned_messages) else "нет",
            ),
            ("Текущий ход", str(getattr(session, "turn", "-"))),
            ("Сообщений в истории", str(len(getattr(session, "history", [])))),
            ("Последний ответ агента", self._preview_text(self._safe_last_text(session)) or "-"),
        ]

        lines = ["# Детали session-resilience case", "", "| Поле | Значение |", "|---|---|"]
        for key, value in rows:
            lines.append(
                f"| {key} | {str(value).replace('|', '/').replace(chr(10), ' ')} |"
            )
        return "\n".join(lines) + "\n"

    def _build_actual(
        self,
        *,
        session,
        planned_messages: list[str],
        extra: list[str] | None = None,
    ) -> list[str]:
        actual = [
            f"Плановых пользовательских сообщений: {len(planned_messages)}",
            f"Текущий номер хода: {getattr(session, 'turn', '-')}",
            f"Сообщений в истории: {len(getattr(session, 'history', []))}",
        ]
        last_text = self._safe_last_text(session)
        if last_text:
            actual.append(f"Последний ответ агента: {self._preview_text(last_text)}")
        if extra:
            actual.extend(extra)
        return actual

    def _attach_case_report(
        self,
        *,
        title: str,
        scenario: str,
        expected: list[str],
        actual: list[str],
        verdict_title: str,
        verdict_text: str,
        case_name: str,
        planned_messages: list[str],
        session,
        findings: list[str],
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
        attach_markdown(
            "turns.md",
            self._turns_markdown(session, planned_messages=planned_messages),
        )
        attach_markdown(
            "case.md",
            self._case_markdown(
                case_name=case_name,
                planned_messages=planned_messages,
                session=session,
            ),
        )
        if findings:
            attach_text("findings.txt", "\n".join(f"- {item}" for item in findings))
        attach_json("[tech] session-resilience-case.json", technical)

    def _run_resilience_case(
        self,
        *,
        session,
        planned_messages: list[str],
        title: str,
        scenario: str,
        case_name: str,
    ) -> None:
        expected = [
            f"Агент корректно отвечает на серию из {len(planned_messages)} сообщений",
            "После серии сообщений probe подтверждает, что сессия остаётся живой",
        ]

        try:
            with step("Прогон серии пользовательских сообщений"):
                run_dialogue(session, planned_messages)

            with step("Проверка, что сессия остаётся работоспособной"):
                session.expect_session_alive()
        except Exception as exc:
            verdict_title, verdict_text = self._classify_failure(exc)
            self._attach_case_report(
                title=title,
                scenario=scenario,
                expected=expected,
                actual=self._build_actual(
                    session=session,
                    planned_messages=planned_messages,
                    extra=[
                        "Проверочный probe: не подтверждён",
                        f"Ошибка: {self._preview_text(str(exc), limit=220)}",
                    ],
                ),
                verdict_title=verdict_title,
                verdict_text=verdict_text,
                case_name=case_name,
                planned_messages=planned_messages,
                session=session,
                findings=[
                    "После серии сообщений сессия не подтвердила живость или проверка завершилась исключением.",
                ],
                technical={
                    "case": case_name,
                    "planned_messages": planned_messages,
                    "turn": getattr(session, "turn", None),
                    "history_length": len(getattr(session, "history", [])),
                    "response_preview": self._preview_text(self._safe_last_text(session), limit=300),
                },
            )
            raise

        self._attach_case_report(
            title=title,
            scenario=scenario,
            expected=expected,
            actual=self._build_actual(
                session=session,
                planned_messages=planned_messages,
                extra=["Проверочный probe: подтверждён"],
            ),
            verdict_title="Проверка пройдена",
            verdict_text="После серии сообщений сессия осталась живой и продолжила принимать запросы.",
            case_name=case_name,
            planned_messages=planned_messages,
            session=session,
            findings=[
                "Сессия выдержала многоходовый сценарий без потери работоспособности.",
            ],
            technical={
                "case": case_name,
                "planned_messages": planned_messages,
                "turn": getattr(session, "turn", None),
                "history_length": len(getattr(session, "history", [])),
                "response_preview": self._preview_text(self._safe_last_text(session), limit=300),
            },
        )

    @pytest.mark.slow
    def test_session_alive_after_5_turns(self, session):
        """Session remains usable after 5 exchanges. / Сессия остаётся работоспособной после 5 обменов."""
        self._run_resilience_case(
            session=session,
            planned_messages=[f"Message {i}" for i in range(1, 6)],
            title="Сессия остаётся живой после 5 сообщений",
            scenario=(
                "Пользователь проходит короткий сценарий из пяти последовательных "
                "сообщений. После этого выполняется probe-проверка живости сессии."
            ),
            case_name="5_turns",
        )

    @pytest.mark.slow
    def test_session_alive_after_10_turns(self, session):
        """Session remains usable after 10 exchanges. / Сессия остаётся работоспособной после 10 обменов."""
        self._run_resilience_case(
            session=session,
            planned_messages=[f"Message {i}" for i in range(1, 11)],
            title="Сессия остаётся живой после 10 сообщений",
            scenario=(
                "Пользователь проходит более длинный сценарий из десяти "
                "последовательных сообщений. После этого выполняется probe-проверка "
                "живости сессии."
            ),
            case_name="10_turns",
        )
