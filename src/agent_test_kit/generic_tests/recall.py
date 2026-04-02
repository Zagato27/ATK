"""
Generic recall tests applicable to any LLM agent.

Thresholds are read from ``agent-test-kit.toml`` → ``[recall]``.

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


class GenericRecallTests:
    """Short- and mid-range fact recall tests.

    Тесты recall для краткосрочного и среднесрочного удержания фактов.
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
                "Проверка не завершилась из-за сетевой или серверной ошибки, "
                "поэтому качество удержания фактов оценить нельзя.",
            )
        if "response text is empty" in lower:
            return (
                "Пустой ответ",
                "Агент формально ответил, но не вернул пользовательского текста, "
                "поэтому recall-сценарий нельзя подтвердить.",
            )
        if lower.startswith("turn") and "http " in lower:
            return (
                "Сервис вернул ошибку",
                "Во время recall-сценария сервис не отдал успешный HTTP-ответ.",
            )
        if "missing keywords" in lower:
            if case == "reference":
                return (
                    "Референсный номер не сохранился",
                    "После нескольких ходов агент не смог воспроизвести ранее "
                    "сообщённый reference number.",
                )
            if case == "account":
                return (
                    "Идентификатор аккаунта потерялся после отвлечения",
                    "После off-topic отклонения агент не смог вернуть основной "
                    "факт и воспроизвести account ID.",
                )
            return (
                "Факт не сохранился в ответе",
                "Ожидаемый факт не найден в финальном ответе агента.",
            )
        if "evaluate_direct(" in lower or ("score=" in lower and "threshold=" in lower):
            if case == "preference":
                return (
                    "Оценка judge ниже порога",
                    "Judge-проверка показала, что агент недостаточно уверенно "
                    "вспомнил пользовательское предпочтение.",
                )
            return (
                "Оценка judge ниже порога",
                "Judge-проверка показала, что агент недостаточно уверенно "
                "удержал несколько фактов из предыдущих сообщений.",
            )
        return (
            "Нарушено ожидаемое поведение",
            "Результат проверки не соответствует ожидаемому удержанию фактов в диалоге.",
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
        expected_facts: list[str],
        intermediate_messages: list[str],
        final_prompt: str,
        threshold: float | None,
        session,
    ) -> str:
        rows = [
            ("Тип кейса", case_type),
            ("Сценарий", case_name),
            ("Ожидаемые факты", "; ".join(expected_facts)),
            ("Количество промежуточных сообщений", str(len(intermediate_messages))),
            ("Финальный запрос", self._preview_text(final_prompt, limit=220) or "-"),
            ("Последний ответ агента", self._preview_text(self._safe_last_text(session)) or "-"),
            ("Длина ответа", str(len(self._safe_last_text(session) or ""))),
            ("Текущий ход", str(getattr(session, "turn", "-"))),
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

        lines = ["# Детали recall-case", "", "| Поле | Значение |", "|---|---|"]
        for key, value in rows:
            safe_value = str(value).replace("|", "/").replace("\n", " ")
            lines.append(f"| {key} | {safe_value} |")
        return "\n".join(lines) + "\n"

    def _build_actual(
        self,
        *,
        session,
        threshold: float | None = None,
        extra: list[str] | None = None,
    ) -> list[str]:
        actual = [
            f"Текущий номер хода: {getattr(session, 'turn', '-')}",
            f"Сообщений в диалоге: {len(getattr(session, 'history', []))}",
        ]
        last_text = self._safe_last_text(session)
        if last_text:
            actual.append(f"Последний ответ агента: {self._preview_text(last_text)}")
        if threshold is not None:
            actual.append(f"Порог judge: {threshold:.2f}")

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
        expected_facts: list[str],
        intermediate_messages: list[str],
        final_prompt: str,
        threshold: float | None,
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
                expected_facts=expected_facts,
                intermediate_messages=intermediate_messages,
                final_prompt=final_prompt,
                threshold=threshold,
                session=session,
            ),
        )
        if findings:
            attach_text("findings.txt", "\n".join(f"- {item}" for item in findings))
        attach_json("[tech] recall-case.json", technical)

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

    def _run_deterministic_case(
        self,
        *,
        session,
        initial_prompt: str,
        intermediate_messages: list[str],
        final_prompt: str,
        expected_fact: str,
        title: str,
        scenario: str,
        case_type: str,
        case_name: str,
        case_key: str,
    ) -> None:
        expected = [
            "Агент корректно отвечает после исходного факта",
            "Финальный ответ содержит ранее сохранённый факт",
        ]

        try:
            with step("Пользователь сообщает факт"):
                session.send(initial_prompt)
                session.expect_response_ok()

            if intermediate_messages:
                with step("Прогон промежуточных сообщений"):
                    run_dialogue(session, intermediate_messages)

            with step("Пользователь просит вспомнить факт"):
                session.send(final_prompt)
                session.expect_response_ok()

            with step("Проверка точного воспроизведения факта"):
                session.expect_contains(expected_fact)
        except Exception as exc:
            verdict_title, verdict_text = self._classify_failure(exc, case=case_key)
            self._attach_case_report(
                title=title,
                scenario=scenario,
                expected=expected,
                verdict_title=verdict_title,
                verdict_text=verdict_text,
                actual=self._build_actual(
                    session=session,
                    extra=[
                        f"Исходный факт: {expected_fact}",
                        f"Финальный запрос: {final_prompt}",
                        f"Ошибка: {self._preview_text(str(exc), limit=220)}",
                    ],
                ),
                findings=[
                    "Финальный ответ не подтвердил, что агент удержал конкретный факт в памяти.",
                ],
                case_type=case_type,
                case_name=case_name,
                expected_facts=[expected_fact],
                intermediate_messages=intermediate_messages,
                final_prompt=final_prompt,
                threshold=None,
                session=session,
                technical={
                    "case": case_key,
                    "initial_prompt": initial_prompt,
                    "intermediate_messages": intermediate_messages,
                    "final_prompt": final_prompt,
                    "expected_fact": expected_fact,
                    "response_preview": self._preview_text(self._safe_last_text(session), limit=300),
                    "response_length": len(self._safe_last_text(session) or ""),
                    "turn": getattr(session, "turn", None),
                },
            )
            raise

        self._attach_case_report(
            title=title,
            scenario=scenario,
            expected=expected,
            verdict_title="Проверка пройдена",
            verdict_text="Агент сохранил факт и корректно воспроизвёл его в финальном ответе.",
            actual=self._build_actual(
                session=session,
                extra=[
                    f"Исходный факт: {expected_fact}",
                    f"Финальный запрос: {final_prompt}",
                ],
            ),
            findings=[
                "Конкретный факт успешно сохраняется и воспроизводится позже в диалоге.",
            ],
            case_type=case_type,
            case_name=case_name,
            expected_facts=[expected_fact],
            intermediate_messages=intermediate_messages,
            final_prompt=final_prompt,
            threshold=None,
            session=session,
            technical={
                "case": case_key,
                "initial_prompt": initial_prompt,
                "intermediate_messages": intermediate_messages,
                "final_prompt": final_prompt,
                "expected_fact": expected_fact,
                "response_preview": self._preview_text(self._safe_last_text(session), limit=300),
                "response_length": len(self._safe_last_text(session) or ""),
                "turn": getattr(session, "turn", None),
            },
        )

    def _run_semantic_case(
        self,
        *,
        session,
        fact_prompts: list[str],
        intermediate_messages: list[str],
        final_prompt: str,
        expected_facts: list[str],
        title: str,
        scenario: str,
        case_type: str,
        case_name: str,
        case_key: str,
        threshold: float,
    ) -> None:
        expected = [
            "Агент корректно отвечает после каждого сохранённого факта",
            "Финальный ответ отражает ранее сообщённую информацию по смыслу",
            f"Judge-оценка по метрике context_retention не ниже {threshold:.2f}",
        ]

        try:
            for idx, prompt in enumerate(fact_prompts, start=1):
                with step(f"Пользователь сообщает факт {idx}"):
                    session.send(prompt)
                    session.expect_response_ok()

            if intermediate_messages:
                with step("Прогон промежуточных сообщений"):
                    run_dialogue(session, intermediate_messages)

            with step("Пользователь просит вспомнить ранее сообщённые факты"):
                session.send(final_prompt)
                session.expect_response_ok()

            with step("Judge проверяет сохранение фактов по смыслу"):
                session.evaluate("context_retention", threshold=threshold)
        except Exception as exc:
            verdict_title, verdict_text = self._classify_failure(exc, case=case_key)
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
                        f"Финальный запрос: {final_prompt}",
                        f"Ожидаемые факты: {', '.join(expected_facts)}",
                        f"Ошибка: {self._preview_text(str(exc), limit=220)}",
                    ],
                ),
                findings=[
                    "Judge или базовая проверка не подтвердили, что агент удержал нужные факты по смыслу.",
                ],
                case_type=case_type,
                case_name=case_name,
                expected_facts=expected_facts,
                intermediate_messages=intermediate_messages,
                final_prompt=final_prompt,
                threshold=threshold,
                session=session,
                technical={
                    "case": case_key,
                    "fact_prompts": fact_prompts,
                    "intermediate_messages": intermediate_messages,
                    "final_prompt": final_prompt,
                    "expected_facts": expected_facts,
                    "threshold": threshold,
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
            verdict_text="Агент удержал факты в памяти, а judge подтвердил корректное воспроизведение по смыслу.",
            actual=self._build_actual(
                session=session,
                threshold=threshold,
                extra=[
                    f"Финальный запрос: {final_prompt}",
                    f"Ожидаемые факты: {', '.join(expected_facts)}",
                ],
            ),
            findings=[
                "Семантическое удержание фактов осталось на ожидаемом уровне.",
            ],
            case_type=case_type,
            case_name=case_name,
            expected_facts=expected_facts,
            intermediate_messages=intermediate_messages,
            final_prompt=final_prompt,
            threshold=threshold,
            session=session,
            technical={
                "case": case_key,
                "fact_prompts": fact_prompts,
                "intermediate_messages": intermediate_messages,
                "final_prompt": final_prompt,
                "expected_facts": expected_facts,
                "threshold": threshold,
                "response_preview": self._preview_text(self._safe_last_text(session), limit=300),
                "response_length": len(self._safe_last_text(session) or ""),
                "turn": getattr(session, "turn", None),
                "eval_result": self._eval_result_data(session),
            },
        )

    def test_reference_number_recalled_deterministically(self, session):
        """Stable reference number is recalled across several turns. / Стабильный референсный номер вспоминается через несколько ходов."""
        self._run_deterministic_case(
            session=session,
            initial_prompt="My reference number is REF-98765",
            intermediate_messages=["Let's continue", "Tell me more"],
            final_prompt="What was my reference number?",
            expected_fact="REF-98765",
            title="Агент помнит reference number через несколько ходов",
            scenario=(
                "Пользователь сообщает reference number, затем продолжает диалог "
                "ещё двумя репликами и после этого просит воспроизвести номер."
            ),
            case_type="Точный recall",
            case_name="reference number после нескольких ходов",
            case_key="reference",
        )

    def test_account_id_survives_off_topic_detour(self, session):
        """Stable account ID survives an off-topic interruption. / Идентификатор аккаунта сохраняется после off-topic отвлечения."""
        self._run_deterministic_case(
            session=session,
            initial_prompt="My account ID is ACC-555",
            intermediate_messages=["By the way, what's the weather like?"],
            final_prompt="OK, back to business. What's my account ID?",
            expected_fact="ACC-555",
            title="Агент сохраняет account ID после off-topic отвлечения",
            scenario=(
                "Пользователь сообщает account ID, затем задаёт отвлекающий "
                "off-topic вопрос и после этого возвращается к основному сценарию."
            ),
            case_type="Точный recall",
            case_name="account ID после off-topic detour",
            case_key="account",
        )

    @pytest.mark.judge
    def test_stated_preference_recalled_semantically(self, session):
        """Agent semantically recalls a user preference. / Агент семантически вспоминает предпочтение пользователя."""
        cfg = get_config().recall
        self._run_semantic_case(
            session=session,
            fact_prompts=["I prefer email communication"],
            intermediate_messages=["Continue", "What should we do next?", "Tell me more"],
            final_prompt="How did I say I prefer to be contacted?",
            expected_facts=["предпочтительный канал связи: email"],
            title="Агент помнит пользовательское предпочтение по смыслу",
            scenario=(
                "Пользователь сообщает предпочтительный способ связи, затем "
                "диалог продолжается несколькими нейтральными сообщениями. "
                "После этого агент должен вспомнить предпочтение по смыслу."
            ),
            case_type="Семантический recall",
            case_name="предпочтительный канал связи",
            case_key="preference",
            threshold=cfg.judge_threshold,
        )

    @pytest.mark.judge
    def test_multiple_facts_retained_semantically(self, session):
        """Agent retains multiple facts provided in separate turns. / Агент удерживает несколько фактов, сообщённых в разных репликах."""
        cfg = get_config().recall
        self._run_semantic_case(
            session=session,
            fact_prompts=[
                "My name is Maria",
                "I am 28 years old",
                "I work as an engineer",
            ],
            intermediate_messages=[],
            final_prompt="Can you summarize what you know about me?",
            expected_facts=["Maria", "28 years old", "engineer"],
            title="Агент удерживает несколько фактов о пользователе",
            scenario=(
                "Пользователь по очереди сообщает несколько фактов о себе. "
                "Затем агенту нужно суммарно воспроизвести эти сведения."
            ),
            case_type="Семантический recall",
            case_name="несколько фактов в отдельных сообщениях",
            case_key="multiple_facts",
            threshold=cfg.judge_threshold,
        )
