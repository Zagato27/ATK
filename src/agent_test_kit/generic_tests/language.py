"""
Generic language/script policy tests applicable to any LLM agent.

Policy is read from ``agent-test-kit.toml`` → ``[language]``.

Requires a ``session`` fixture returning an initialized
:class:`~agent_test_kit.AgentSession`.
"""
from __future__ import annotations

import re
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


class GenericLanguageTests:
    """Language/script policy assertions for any LLM agent."""

    _INPUT_CASES = {
        "latin": "англоязычный ввод",
        "cyrillic": "русскоязычный ввод",
        "arabic": "арабский ввод",
        "cjk": "CJK-ввод",
    }
    _SCRIPT_PATTERNS = {
        "latin": re.compile(r"[A-Za-z]"),
        "cyrillic": re.compile(r"[А-Яа-яЁё]"),
        "arabic": re.compile(r"[\u0600-\u06FF]"),
        "cjk": re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]"),
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

    def _classify_failure(self, exc: Exception) -> tuple[str, str]:
        message = str(exc)
        lower = message.lower()

        if self._looks_like_environment_error(message):
            return (
                "Проблема окружения или доступности сервиса",
                "Проверка не дошла до оценки языковой политики, потому что "
                "запрос завершился сетевой или серверной ошибкой.",
            )
        if "response text is empty" in lower:
            return (
                "Пустой ответ",
                "Агент формально ответил, но не вернул пользовательского текста, "
                "поэтому соблюдение языковой политики нельзя подтвердить.",
            )
        if lower.startswith("turn") and "http " in lower:
            return (
                "Сервис вернул ошибку",
                "Политику скрипта нельзя оценить, потому что сервис не вернул "
                "успешный HTTP-ответ.",
            )
        if "expected script" in lower and "not found" in lower:
            return (
                "Ожидаемый скрипт не найден в ответе",
                "Ответ агента не содержит символов того скрипта, который "
                "ожидался по текущей языковой политике.",
            )
        if "ratio" in lower and "required" in lower:
            return (
                "Доля ожидаемого скрипта слишком мала",
                "Нужный скрипт присутствует, но занимает слишком малую часть ответа, "
                "поэтому политика языка считается нарушенной.",
            )
        if "unsupported script" in lower or "unsupported language.mode" in lower:
            return (
                "Некорректная конфигурация языковой политики",
                "Текущая конфигурация language использует неподдерживаемый режим "
                "или скрипт.",
            )
        return (
            "Нарушено ожидаемое поведение",
            "Результат сценария не соответствует ожидаемой языковой политике.",
        )

    def _counts_markdown(
        self,
        *,
        prompt: str,
        input_script: str,
        expected_script: str,
        counts: dict[str, int],
        ratio: float,
        required_ratio: float,
        session,
        mode: str,
        fixed_script: str,
    ) -> str:
        last_text = self._safe_last_text(session)
        total = sum(counts.values())
        lines = [
            "# Детали language-case",
            "",
            "| Поле | Значение |",
            "|---|---|",
            f"| Режим language.mode | {mode} |",
            f"| fixed_script | {fixed_script} |",
            f"| Входной скрипт | {input_script} |",
            f"| Ожидаемый скрипт | {expected_script} |",
            f"| Минимальная доля ожидаемого скрипта | {required_ratio:.2f} |",
            f"| Фактическая доля ожидаемого скрипта | {ratio:.2f} |",
            f"| Сумма распознанных символов | {total} |",
            f"| Превью запроса | {(self._preview_text(prompt) or repr(prompt)).replace('|', '/')} |",
            f"| Превью ответа | {(self._preview_text(last_text) or '-').replace('|', '/')} |",
            "",
            "## Распределение по скриптам",
            "",
            "| Скрипт | Количество символов |",
            "|---|---:|",
        ]
        for script, count in counts.items():
            lines.append(f"| {script} | {count} |")
        return "\n".join(lines) + "\n"

    def _build_actual(
        self,
        session,
        *,
        prompt: str,
        input_script: str,
        expected_script: str,
        counts: dict[str, int],
        ratio: float,
        required_ratio: float,
        extra: list[str] | None = None,
    ) -> list[str]:
        actual = [
            f"Текущий номер хода: {getattr(session, 'turn', '-')}",
            f"Входной скрипт: {input_script}",
            f"Ожидаемый скрипт: {expected_script}",
            f"Минимально допустимая доля ожидаемого скрипта: {required_ratio:.2f}",
            f"Фактическая доля ожидаемого скрипта: {ratio:.2f}",
            f"Распределение символов по скриптам: {counts}",
            f"Превью запроса: {self._preview_text(prompt) or repr(prompt)}",
        ]
        last_text = self._safe_last_text(session)
        if last_text is not None:
            actual.append(
                f"Превью ответа: {self._preview_text(last_text) or repr(last_text)}"
            )
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
        prompt: str,
        input_script: str,
        expected_script: str,
        counts: dict[str, int],
        ratio: float,
        required_ratio: float,
        verdict_title: str,
        verdict_text: str,
        actual_extra: list[str] | None = None,
        findings: list[str] | None = None,
        technical: dict[str, Any] | None = None,
        mode: str,
        fixed_script: str,
    ) -> None:
        actual = self._build_actual(
            session,
            prompt=prompt,
            input_script=input_script,
            expected_script=expected_script,
            counts=counts,
            ratio=ratio,
            required_ratio=required_ratio,
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
            "scripts.md",
            self._counts_markdown(
                prompt=prompt,
                input_script=input_script,
                expected_script=expected_script,
                counts=counts,
                ratio=ratio,
                required_ratio=required_ratio,
                session=session,
                mode=mode,
                fixed_script=fixed_script,
            ),
        )
        if findings:
            attach_text("findings.txt", "\n".join(f"- {item}" for item in findings))
        if technical is not None:
            attach_json("[tech] language-case.json", technical)

    @pytest.mark.parametrize(
        ("prompt", "input_script"),
        [
            pytest.param("Hello, can you help me?", "latin", id="english_input"),
            pytest.param("Здравствуйте, помогите, пожалуйста.", "cyrillic", id="russian_input"),
        ],
    )
    def test_language_script_policy(self, session, prompt, input_script):
        """Response follows the configured script policy. / Ответ соблюдает настроенную политику скрипта."""
        cfg = get_config().language
        case_name = self._INPUT_CASES.get(input_script, input_script)
        title = f"Политика скрипта: {case_name}"
        if cfg.mode == "ignore":
            summary = (
                "Сценарий:\n"
                "Проверка политики языка для ответа агента.\n\n"
                "Ожидалось:\n"
                "- Для ответа применяется настроенная политика скрипта.\n\n"
                "Фактически:\n"
                "- Проверка пропущена, потому что `language.mode=ignore`.\n\n"
                "Вывод:\n"
                "Проверка пропущена. Языковая политика явно отключена в конфигурации."
            )
            set_title(title)
            set_description(summary)
            attach_text("summary.txt", summary)
            pytest.skip("language.mode=ignore")

        expected_script = self._resolve_expected_script(
            mode=cfg.mode,
            input_script=input_script,
            fixed_script=cfg.fixed_script,
        )
        if expected_script not in self._SCRIPT_PATTERNS:
            raise ValueError(
                f"Unsupported script '{expected_script}'. "
                f"Supported: {sorted(self._SCRIPT_PATTERNS)}"
            )
        scenario = (
            "Пользователь отправляет сообщение на определённом скрипте. "
            "Нужно убедиться, что ответ агента следует настроенной политике "
            "языка и скрипта."
        )
        expected = [
            "Агент возвращает непустой ответ",
            f"В ответе присутствует ожидаемый скрипт `{expected_script}`",
            f"Доля ожидаемого скрипта не ниже {cfg.min_expected_script_ratio:.2f}",
        ]
        counts = {script: 0 for script in self._SCRIPT_PATTERNS}
        ratio = 0.0

        try:
            with step("Определение ожидаемого скрипта по конфигурации"):
                resolved_expected_script = expected_script

            with step("Отправка пользовательского запроса"):
                session.send(prompt)

            with step("Проверка базовой корректности ответа"):
                session.expect_response_ok()

            with step("Подсчёт символов по скриптам"):
                counts = self._script_counts(session.last_text)
                total = sum(counts.values())
                expected_count = counts.get(resolved_expected_script, 0)
                ratio = expected_count / total if total else 0.0

            with step("Проверка наличия ожидаемого скрипта"):
                assert expected_count > 0, (
                    f"Expected script '{resolved_expected_script}' not found in response. "
                    f"Counts: {counts}. Response: {session.last_text[:300]!r}"
                )

            with step("Проверка минимальной доли ожидаемого скрипта"):
                assert ratio >= cfg.min_expected_script_ratio, (
                    f"Expected script '{resolved_expected_script}' ratio {ratio:.2f} "
                    f"< required {cfg.min_expected_script_ratio:.2f}. "
                    f"Counts: {counts}. Response: {session.last_text[:300]!r}"
                )
        except Exception as exc:
            verdict_title, verdict_text = self._classify_failure(exc)
            self._attach_case_report(
                title=title,
                scenario=scenario,
                expected=expected,
                session=session,
                prompt=prompt,
                input_script=input_script,
                expected_script=expected_script,
                counts=counts,
                ratio=ratio,
                required_ratio=cfg.min_expected_script_ratio,
                verdict_title=verdict_title,
                verdict_text=verdict_text,
                actual_extra=[
                    f"Режим language.mode: {cfg.mode}",
                    f"fixed_script: {cfg.fixed_script}",
                    f"Ошибка: {self._preview_text(str(exc), limit=220)}",
                ],
                findings=[
                    "Ответ не подтвердил настроенную языковую политику."
                ],
                technical={
                    "case": "language_script_policy",
                    "prompt": prompt,
                    "input_script": input_script,
                    "expected_script": expected_script,
                    "mode": cfg.mode,
                    "fixed_script": cfg.fixed_script,
                    "min_expected_script_ratio": cfg.min_expected_script_ratio,
                    "counts": counts,
                    "ratio": ratio,
                    "turn": getattr(session, "turn", None),
                },
                mode=cfg.mode,
                fixed_script=cfg.fixed_script,
            )
            raise

        self._attach_case_report(
            title=title,
            scenario=scenario,
            expected=expected,
            session=session,
            prompt=prompt,
            input_script=input_script,
            expected_script=expected_script,
            counts=counts,
            ratio=ratio,
            required_ratio=cfg.min_expected_script_ratio,
            verdict_title="Проверка пройдена",
            verdict_text=(
                "Ответ агента соответствует настроенной политике скрипта и "
                "содержит достаточную долю ожидаемых символов."
            ),
            actual_extra=[
                f"Режим language.mode: {cfg.mode}",
                f"fixed_script: {cfg.fixed_script}",
            ],
            findings=[
                "Языковая политика ответа соблюдена.",
            ],
            technical={
                "case": "language_script_policy",
                "prompt": prompt,
                "input_script": input_script,
                "expected_script": expected_script,
                "mode": cfg.mode,
                "fixed_script": cfg.fixed_script,
                "min_expected_script_ratio": cfg.min_expected_script_ratio,
                "counts": counts,
                "ratio": ratio,
                "turn": getattr(session, "turn", None),
            },
            mode=cfg.mode,
            fixed_script=cfg.fixed_script,
        )

    @classmethod
    def _script_counts(cls, text: str) -> dict[str, int]:
        return {
            script: len(pattern.findall(text))
            for script, pattern in cls._SCRIPT_PATTERNS.items()
        }

    @staticmethod
    def _resolve_expected_script(
        *,
        mode: str,
        input_script: str,
        fixed_script: str,
    ) -> str:
        if mode == "match_input":
            return input_script
        if mode == "fixed":
            return fixed_script
        raise ValueError(
            f"Unsupported language.mode '{mode}'. "
            "Supported: ignore, match_input, fixed"
        )
