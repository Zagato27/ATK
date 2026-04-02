"""
Generic correction-handling tests applicable to any LLM agent.

Thresholds are read from ``agent-test-kit.toml`` → ``[corrections]``.

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


class GenericCorrectionTests:
    """User-correction handling tests.

    Тесты обработки исправлений от пользователя.
    """

    @staticmethod
    def _preview_text(text: str | None, limit: int = 160) -> str | None:
        if text is None:
            return None
        compact = " ".join(text.split())
        if len(compact) <= limit:
            return compact
        return compact[: limit - 3] + "..."

    def _safe_last_text(self, session) -> str | None:
        try:
            return self._preview_text(session.last_text)
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
                "Проверка не дошла до оценки бизнес-поведения, потому что запрос "
                "завершился сетевой или серверной ошибкой.",
            )

        if case == "judge":
            return (
                "Оценка judge ниже порога",
                "Семантическая проверка показала, что агент недостаточно уверенно "
                "принял исправление и продолжил сценарий.",
            )

        if case == "city" and ("forbidden" in lower or "none of" in lower):
            return (
                "Исправление не закрепилось в памяти",
                "После замены города агент либо сохранил старое значение, либо "
                "не подтвердил новое.",
            )

        if case == "name":
            return (
                "Агент повторно запрашивает уже известное имя",
                "После того как пользователь уже сообщил имя, агент не должен "
                "спрашивать его повторно распространёнными формулировками.",
            )

        return (
            "Нарушено ожидаемое поведение",
            "Результат сценария не соответствует ожидаемому поведению агента.",
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

    def _build_actual(
        self,
        session,
        *,
        extra: list[str] | None = None,
    ) -> list[str]:
        actual = [
            f"Текущий номер хода: {getattr(session, 'turn', '-')}",
            f"Сообщений в диалоге: {len(getattr(session, 'history', []))}",
        ]
        last_text = self._safe_last_text(session)
        if last_text:
            actual.append(f"Последний ответ агента: {last_text}")

        result = getattr(session, "last_eval_result", None)
        if result is not None:
            actual.append(
                f"Judge score: {getattr(result, 'score', 0.0):.3f}"
            )
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
        verdict_title: str,
        verdict_text: str,
        actual_extra: list[str] | None = None,
        findings: list[str] | None = None,
        technical: dict[str, Any] | None = None,
    ) -> None:
        actual = self._build_actual(session, extra=actual_extra)
        summary_lines = [
            f"Сценарий:\n{scenario}",
            "",
            "Ожидалось:",
        ]
        summary_lines.extend(f"- {item}" for item in expected)
        summary_lines.extend(["", "Фактически:"])
        summary_lines.extend(f"- {item}" for item in actual)
        summary_lines.extend(
            [
                "",
                f"Вывод:\n{verdict_title}. {verdict_text}",
            ]
        )
        summary = "\n".join(summary_lines)

        set_title(title)
        set_description(summary)
        attach_text("summary.txt", summary)
        attach_markdown("dialogue.md", self._dialogue_markdown(session))
        if findings:
            attach_text("findings.txt", "\n".join(f"- {item}" for item in findings))
        if technical is not None:
            attach_json("[tech] correction-case.json", technical)

    @pytest.mark.judge
    def test_accepts_correction_semantically(self, session):
        """Agent accepts and processes a correction. / Агент принимает и обрабатывает исправление."""
        cfg = get_config().corrections
        title = "Агент принимает исправление возраста"
        scenario = (
            "Пользователь сначала сообщает возраст, затем сразу исправляет его. "
            "Нужно убедиться, что агент принимает исправление и продолжает "
            "сценарий без конфликта."
        )
        expected = [
            "Агент корректно отвечает на исходное сообщение",
            "После исправления агент принимает новое значение без спора",
            f"Judge-оценка по метрике correction_handling не ниже {cfg.judge_threshold:.2f}",
        ]

        try:
            with step("Пользователь сообщает исходный возраст"):
                session.send("My age is 30")
                session.expect_response_ok()

            with step("Пользователь исправляет возраст"):
                session.send("Sorry, I made a mistake — I'm actually 31")
                session.expect_response_ok()

            with step("Judge проверяет обработку исправления"):
                session.evaluate("correction_handling", threshold=cfg.judge_threshold)
        except Exception as exc:
            verdict_title, verdict_text = self._classify_failure(exc, case="judge")
            self._attach_case_report(
                title=title,
                scenario=scenario,
                expected=expected,
                session=session,
                verdict_title=verdict_title,
                verdict_text=verdict_text,
                actual_extra=[
                    f"Порог judge: {cfg.judge_threshold:.2f}",
                    f"Ошибка: {self._preview_text(str(exc), limit=220)}",
                ],
                findings=[
                    "Проверка завершилась исключением до подтверждения успешной обработки исправления.",
                ],
                technical={
                    "case": "accepts_correction_semantically",
                    "metric": "correction_handling",
                    "threshold": cfg.judge_threshold,
                    "turn": session.turn,
                },
            )
            raise

        self._attach_case_report(
            title=title,
            scenario=scenario,
            expected=expected,
            session=session,
            verdict_title="Проверка пройдена",
            verdict_text=(
                "Агент принял исправление, не потерял контекст и успешно прошёл "
                "семантическую проверку обработки исправления."
            ),
            actual_extra=[
                f"Порог judge: {cfg.judge_threshold:.2f}",
                "Семантическая проверка завершилась без исключений.",
            ],
            findings=[
                "Исправление было обработано в рамках ожидаемого сценария.",
            ],
            technical={
                "case": "accepts_correction_semantically",
                "metric": "correction_handling",
                "threshold": cfg.judge_threshold,
                "turn": session.turn,
            },
        )

    def test_corrected_city_overrides_previous_value(self, session):
        """Corrected city replaces the old one in later recall. / Исправленный город заменяет старое значение при последующем recall."""
        title = "Исправленный город заменяет старое значение"
        scenario = (
            "Пользователь сначала сообщает один город проживания, затем "
            "исправляет его и просит подтвердить актуальный город."
        )
        expected = [
            "Агент корректно отвечает на оба пользовательских сообщения",
            "В финальном ответе фигурирует новый город Berlin",
            "Старый город London больше не используется",
        ]

        try:
            with step("Пользователь сообщает исходный город"):
                session.send("I live in London")
                session.expect_response_ok()

            with step("Пользователь исправляет город"):
                session.send("Actually, I moved — I live in Berlin now")
                session.expect_response_ok()

            with step("Пользователь просит подтвердить актуальный город"):
                session.send("Can you confirm where I live?")
                session.expect_response_ok()

            with step("Проверка, что новый город сохранён"):
                session.expect_contains_any("berlin", "Berlin")

            with step("Проверка, что старый город больше не упоминается"):
                session.expect_not_contains("london", "London")
        except Exception as exc:
            verdict_title, verdict_text = self._classify_failure(exc, case="city")
            self._attach_case_report(
                title=title,
                scenario=scenario,
                expected=expected,
                session=session,
                verdict_title=verdict_title,
                verdict_text=verdict_text,
                actual_extra=[
                    "Ожидали увидеть Berlin и не увидеть London.",
                    f"Ошибка: {self._preview_text(str(exc), limit=220)}",
                ],
                findings=[
                    "Проверка recall после исправления не подтвердила корректную замену значения.",
                ],
                technical={
                    "case": "corrected_city_overrides_previous_value",
                    "expected_new_value": "Berlin",
                    "forbidden_old_value": "London",
                    "turn": session.turn,
                },
            )
            raise

        self._attach_case_report(
            title=title,
            scenario=scenario,
            expected=expected,
            session=session,
            verdict_title="Проверка пройдена",
            verdict_text=(
                "После исправления агент закрепил новое значение и больше "
                "не использовал старый город."
            ),
            actual_extra=[
                "Финальный recall подтвердил Berlin без упоминания London.",
            ],
            findings=[
                "Исправление значения успешно сохранилось в последующем recall.",
            ],
            technical={
                "case": "corrected_city_overrides_previous_value",
                "expected_new_value": "Berlin",
                "forbidden_old_value": "London",
                "turn": session.turn,
            },
        )

    def test_answered_name_not_reasked_with_common_phrasings(self, session):
        """Agent does not re-ask the already provided name using common phrasings. / Агент не переспрашивает уже сообщённое имя распространёнными формулировками."""
        title = "Агент не спрашивает имя повторно"
        scenario = (
            "Пользователь сообщает своё имя и затем просит продолжить диалог. "
            "Нужно убедиться, что агент не переспрашивает уже известное имя."
        )
        expected = [
            "Агент корректно отвечает после сообщения имени",
            "При продолжении сценария агент не спрашивает имя заново",
            "Во втором ответе отсутствуют типовые фразы повторного запроса имени",
        ]
        forbidden_phrases = [
            "your name",
            "what is your name",
            "ваше имя",
            "как вас зовут",
        ]
        matched_phrases: list[str] = []

        try:
            with step("Пользователь сообщает имя"):
                session.send("My name is Alexander")
                session.expect_response_ok()

            with step("Пользователь просит продолжить сценарий"):
                session.send("Let's continue")
                session.expect_response_ok()

            with step("Проверка, что имя не запрашивается повторно"):
                second_response = session.last_text.lower()
                matched_phrases = [
                    phrase for phrase in forbidden_phrases if phrase in second_response
                ]
                assert not matched_phrases, (
                    f"Agent re-asked for name in: {session.last_text[:300]}"
                )
        except Exception as exc:
            verdict_title, verdict_text = self._classify_failure(exc, case="name")
            extra = [
                f"Проверяемые фразы: {', '.join(forbidden_phrases)}",
                f"Ошибка: {self._preview_text(str(exc), limit=220)}",
            ]
            if matched_phrases:
                extra.append(
                    f"Найдены запрещённые фразы: {', '.join(matched_phrases)}"
                )
            self._attach_case_report(
                title=title,
                scenario=scenario,
                expected=expected,
                session=session,
                verdict_title=verdict_title,
                verdict_text=verdict_text,
                actual_extra=extra,
                findings=[
                    "Во втором ответе обнаружены признаки повторного запроса уже известного имени."
                    if matched_phrases
                    else "Проверка сценария завершилась исключением до успешного подтверждения поведения.",
                ],
                technical={
                    "case": "answered_name_not_reasked_with_common_phrasings",
                    "forbidden_phrases": forbidden_phrases,
                    "matched_phrases": matched_phrases,
                    "turn": session.turn,
                },
            )
            raise

        self._attach_case_report(
            title=title,
            scenario=scenario,
            expected=expected,
            session=session,
            verdict_title="Проверка пройдена",
            verdict_text=(
                "Агент запомнил уже сообщённое имя и продолжил диалог без "
                "лишнего повторного запроса."
            ),
            actual_extra=[
                f"Проверяемые фразы: {', '.join(forbidden_phrases)}",
                "Запрещённые формулировки повторного запроса имени не обнаружены.",
            ],
            findings=[
                "Имя пользователя не было повторно запрошено во втором ответе.",
            ],
            technical={
                "case": "answered_name_not_reasked_with_common_phrasings",
                "forbidden_phrases": forbidden_phrases,
                "matched_phrases": matched_phrases,
                "turn": session.turn,
            },
        )
