"""
Generic scope-recovery tests applicable to any LLM agent.

Thresholds are read from ``agent-test-kit.toml`` -> ``[scope_recovery]``.

Requires a ``session`` fixture returning an initialized
:class:`~agent_test_kit.AgentSession`.

Универсальные тесты возврата к основному сценарию для любого LLM-агента.
Пороги берутся из ``agent-test-kit.toml`` -> ``[scope_recovery]``.
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


def _dialogue_context(session) -> list[str]:
    """Materialize earlier turns for judge-based multi-turn checks."""
    return [f"{item['role']}: {item['content']}" for item in session.history[:-2]]


class GenericScopeRecoveryTests:
    """Recovery tests after off-topic detours.

    Тесты возврата к основному сценарию после отвлечения от темы.
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

    def _detected_forbidden(
        self,
        session,
        forbidden: list[str] | tuple[str, ...],
    ) -> list[str]:
        last_text = self._safe_last_text(session)
        if not last_text:
            return []
        lower = last_text.lower()
        return [item for item in forbidden if item.lower() in lower]

    def _classify_failure(
        self,
        exc: Exception,
        *,
        forbidden_hits: list[str],
    ) -> tuple[str, str]:
        message = str(exc)
        lower = message.lower()

        if self._looks_like_environment_error(message):
            return (
                "Проблема окружения или доступности сервиса",
                "Проверка не дошла до оценки возврата к основному сценарию, "
                "потому что запрос завершился сетевой или серверной ошибкой.",
            )
        if "response text is empty" in lower:
            return (
                "Пустой ответ",
                "Агент формально ответил, но не вернул пользовательского текста, "
                "поэтому возврат к сценарию нельзя подтвердить.",
            )
        if lower.startswith("turn") and "http " in lower:
            return (
                "Сервис вернул ошибку",
                "Сценарий возврата к основному потоку нельзя оценить, потому что "
                "сервис не вернул успешный HTTP-ответ.",
            )
        if forbidden_hits:
            return (
                "Агент продолжил off-topic линию",
                "После команды вернуться к основному сценарию ответ всё ещё "
                "содержит off-topic содержание вместо возврата к задаче.",
            )
        return (
            "Оценка judge ниже порога",
            "Judge-проверка показала, что агент недостаточно уверенно вернулся "
            "к основному сценарию после отвлечения.",
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
        prompts: list[str],
        threshold: float,
        session,
        context_used: bool,
        case_type: str,
        forbidden: list[str] | tuple[str, ...],
    ) -> str:
        last_text = self._safe_last_text(session)
        rows = [
            ("Тип кейса", case_type),
            ("Количество пользовательских сообщений", str(len(prompts))),
            ("Порог judge", f"{threshold:.2f}"),
            ("Контекст передан judge", "да" if context_used else "нет"),
            ("Первое сообщение", self._preview_text(prompts[0]) or repr(prompts[0])),
            ("Финальное сообщение", self._preview_text(prompts[-1]) or repr(prompts[-1])),
            ("Off-topic маркеры для контроля", ", ".join(forbidden) if forbidden else "-"),
            ("Последний ответ агента", self._preview_text(last_text) or "-"),
        ]
        result = getattr(session, "last_eval_result", None)
        if result is not None:
            rows.extend(
                [
                    ("Judge score", f"{getattr(result, 'score', 0.0):.3f}"),
                    ("Judge passed", "да" if getattr(result, "passed", False) else "нет"),
                ]
            )

        lines = ["# Детали scope-recovery case", "", "| Поле | Значение |", "|---|---|"]
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
        prompts: list[str],
        threshold: float,
        verdict_title: str,
        verdict_text: str,
        case_type: str,
        context_used: bool,
        forbidden: list[str] | tuple[str, ...],
        actual_extra: list[str] | None = None,
        findings: list[str] | None = None,
        technical: dict[str, Any] | None = None,
    ) -> None:
        actual = self._build_actual(
            session=session,
            threshold=threshold,
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
                prompts=prompts,
                threshold=threshold,
                session=session,
                context_used=context_used,
                case_type=case_type,
                forbidden=forbidden,
            ),
        )
        if findings:
            attach_text("findings.txt", "\n".join(f"- {item}" for item in findings))
        if technical is not None:
            attach_json("[tech] scope-recovery-case.json", technical)

    def _run_scope_recovery_case(
        self,
        *,
        session,
        prompts: list[str],
        threshold: float,
        title: str,
        scenario: str,
        case_type: str,
        expected: list[str],
        forbidden: list[str] | tuple[str, ...] = (),
    ) -> None:
        try:
            for idx, prompt in enumerate(prompts, start=1):
                with step(f"Отправка сообщения {idx}"):
                    session.send(prompt)
                with step(f"Проверка базовой корректности ответа {idx}"):
                    session.expect_response_ok()

            if forbidden:
                with step("Проверка, что агент не продолжает off-topic тему"):
                    session.expect_not_contains(*forbidden)

            with step("Judge проверяет возврат к основному сценарию"):
                session.evaluate(
                    "scope_recovery",
                    threshold=threshold,
                    context=_dialogue_context(session),
                )
        except Exception as exc:
            forbidden_hits = self._detected_forbidden(session, forbidden)
            verdict_title, verdict_text = self._classify_failure(
                exc,
                forbidden_hits=forbidden_hits,
            )
            self._attach_case_report(
                title=title,
                scenario=scenario,
                expected=expected,
                session=session,
                prompts=prompts,
                threshold=threshold,
                verdict_title=verdict_title,
                verdict_text=verdict_text,
                case_type=case_type,
                context_used=True,
                forbidden=forbidden,
                actual_extra=[
                    (
                        "Обнаруженные off-topic маркеры: " + ", ".join(forbidden_hits)
                        if forbidden_hits
                        else "Обнаруженные off-topic маркеры: нет"
                    ),
                    f"Ошибка: {self._preview_text(str(exc), limit=220)}",
                ],
                findings=[
                    "Агент не подтвердил возврат к основному сценарию или проверка завершилась ошибкой.",
                ],
                technical={
                    "case": case_type,
                    "prompts": prompts,
                    "forbidden": list(forbidden),
                    "forbidden_hits": forbidden_hits,
                    "turn": getattr(session, "turn", None),
                    "eval_result": {
                        "score": getattr(getattr(session, "last_eval_result", None), "score", None),
                        "passed": getattr(getattr(session, "last_eval_result", None), "passed", None),
                        "reasoning": getattr(getattr(session, "last_eval_result", None), "reasoning", None),
                    } if getattr(session, "last_eval_result", None) is not None else None,
                },
            )
            raise

        self._attach_case_report(
            title=title,
            scenario=scenario,
            expected=expected,
            session=session,
            prompts=prompts,
            threshold=threshold,
            verdict_title="Проверка пройдена",
            verdict_text=(
                "Агент корректно вернулся к основному сценарию и не продолжил "
                "развивать отвлекающую тему."
            ),
            case_type=case_type,
            context_used=True,
            forbidden=forbidden,
            actual_extra=[
                (
                    "Обнаруженные off-topic маркеры: нет"
                    if forbidden
                    else "Дополнительная проверка off-topic маркеров не требовалась"
                ),
            ],
            findings=[
                "После detour агент удержал основной сценарий и продолжил основной поток.",
            ],
            technical={
                "case": case_type,
                "prompts": prompts,
                "forbidden": list(forbidden),
                "forbidden_hits": [],
                "turn": getattr(session, "turn", None),
                "eval_result": {
                    "score": getattr(getattr(session, "last_eval_result", None), "score", None),
                    "passed": getattr(getattr(session, "last_eval_result", None), "passed", None),
                    "reasoning": getattr(getattr(session, "last_eval_result", None), "reasoning", None),
                } if getattr(session, "last_eval_result", None) is not None else None,
            },
        )

    @pytest.mark.judge
    def test_returns_to_scenario_after_single_detour(self, session):
        """Agent returns to the main scenario after one detour. / Агент возвращается к основному сценарию после одного отвлечения."""
        cfg = get_config().scope_recovery
        self._run_scope_recovery_case(
            session=session,
            prompts=[
                "Hello, I need your help",
                "By the way, what's the capital of France?",
                "OK, let's continue with what we were doing",
            ],
            threshold=cfg.judge_threshold,
            title="Агент возвращается к сценарию после одного отвлечения",
            scenario=(
                "Пользователь начинает основной сценарий, затем уходит в "
                "off-topic вопрос и после этого просит вернуться к исходной задаче."
            ),
            case_type="одно отвлечение",
            expected=[
                "Агент корректно отвечает на все сообщения",
                "После возврата к сценарию финальный ответ не продолжает обсуждать Paris",
                f"Judge подтверждает возврат к основной задаче с порогом не ниже {cfg.judge_threshold:.2f}",
            ],
            forbidden=("Paris",),
        )

    @pytest.mark.judge
    def test_returns_to_scenario_after_repeated_detours(self, session):
        """Agent recovers the main scenario after multiple detours. / Агент возвращает основной сценарий после нескольких отвлечений."""
        cfg = get_config().scope_recovery
        self._run_scope_recovery_case(
            session=session,
            prompts=[
                "I want to start the process",
                "Actually, tell me a joke",
                "And what's the weather tomorrow?",
                "Back to the process: what do you need from me next?",
            ],
            threshold=cfg.judge_threshold,
            title="Агент возвращает основной сценарий после нескольких отвлечений",
            scenario=(
                "Пользователь дважды уводит диалог в сторону, а затем просит "
                "снова продолжить основной процесс."
            ),
            case_type="повторные отвлечения",
            expected=[
                "Агент корректно отвечает на все сообщения",
                "После нескольких detour-реплик агент возвращается к основному процессу",
                f"Judge подтверждает возврат к сценарию с порогом не ниже {cfg.judge_threshold:.2f}",
            ],
        )
