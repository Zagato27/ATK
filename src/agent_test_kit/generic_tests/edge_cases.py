"""
Generic edge-case robustness tests applicable to any LLM agent.

Requires a ``session`` fixture returning an initialized
:class:`~agent_test_kit.AgentSession`.

Универсальные тесты устойчивости к граничным случаям, применимые к любому
LLM-агенту. Требуется фикстура ``session``, возвращающая инициализированный
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


class GenericEdgeCaseTests:
    """Unusual or extreme inputs that should not break the agent.

    Эти тесты intentionally проверяют robustness, а не качество семантической
    обработки. Странный, редкий или экстремальный ввод не должен приводить к
    ошибкам, пустым ответам или явным техническим артефактам.
    """

    _ERROR_ARTIFACTS = (
        "traceback",
        "exception",
        "stack trace",
        "internal server error",
    )
    _MAX_EDGE_RESPONSE_CHARS = 10000
    _MINIMAL_CASES = {
        "": "пустая строка",
        "   ": "только пробелы",
        ".": "одна точка",
        "\n": "перевод строки",
        "\t": "табуляция",
    }
    _UNICODE_CASES = {
        "👍🔥🎉💯🚀" * 10: "только emoji",
        "Hello 👋 I need help 🆘 please": "emoji и обычный текст",
        "α β γ δ ε — « » © ® ™ ¥ € £ ∞ ≠ ≈": "unicode-символы",
        "مرحبا، أحتاج مساعدة": "RTL-текст",
        "你好，我需要帮助。こんにちは。": "CJK-текст",
        "Hello, привет, Hallo, 你好, مرحبا!": "смешанные языки",
    }
    _LONG_CASES = {
        "word " * 1000: "очень длинное сообщение",
        "a" * 2000: "одно очень длинное слово",
        "?" * 500: "повторяющийся символ",
        "1234567890": "только числа",
        "-100 0.5 3.14159 1e10 -0": "разные числовые форматы",
    }
    _STRUCTURED_CASES = {
        '{"key": "value", "nested": {"a": [1,2,3]}}': "JSON-подобный ввод",
        "# Header\n**bold** _italic_ `code` [link](http://x.com)": "markdown-разметка",
        "&lt;tag&gt; &amp; &quot;value&quot; &#x27;": "xml/html-entities",
        "path\\to\\file\\n\\t\\r\\0": "escape-последовательности",
        "before\\x00after": "маркер нулевого байта",
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

    def _detected_artifacts(self, session) -> list[str]:
        last_text = self._safe_last_text(session)
        if not last_text:
            return []
        lower = last_text.lower()
        return [artifact for artifact in self._ERROR_ARTIFACTS if artifact in lower]

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
                "Проверка не дошла до оценки устойчивости, потому что запрос "
                "завершился сетевой или серверной ошибкой.",
            )
        if "response text is empty" in lower:
            return (
                "Пустой ответ на граничный ввод",
                "Агент формально ответил, но не вернул полезного текстового содержания.",
            )
        if "contains forbidden" in lower:
            return (
                "В ответе появились технические артефакты",
                "Вместо устойчивой обработки граничного ввода агент показал "
                "следы внутренней ошибки или технические сообщения.",
            )
        if "length " in lower and "expected [" in lower:
            return (
                "Ответ вышел за допустимую длину",
                "На длинный или вырожденный ввод агент ответил слишком объёмно, "
                "что может указывать на потерю контроля над форматом ответа.",
            )
        if lower.startswith("turn") and "http " in lower:
            return (
                "Сервис вернул ошибку",
                "Граничный кейс не был обработан успешно из-за неуспешного HTTP-ответа.",
            )
        return (
            "Нарушено ожидаемое поведение",
            "Результат сценария не соответствует ожидаемой устойчивости агента.",
        )

    def _resolve_case_name(self, group: str, text: str) -> str:
        mapping = {
            "minimal": self._MINIMAL_CASES,
            "unicode": self._UNICODE_CASES,
            "long": self._LONG_CASES,
            "structured": self._STRUCTURED_CASES,
        }[group]
        if text in mapping:
            return mapping[text]
        fallbacks = {
            "minimal": "минимальный ввод",
            "unicode": "unicode-ввод",
            "long": "длинный или вырожденный ввод",
            "structured": "структурированный или escape-ввод",
        }
        return fallbacks[group]

    def _case_markdown(
        self,
        *,
        category: str,
        case_name: str,
        text: str,
        session,
        max_chars: int | None = None,
    ) -> str:
        last_text = self._safe_last_text(session)
        response_preview = self._preview_text(last_text) or "-"
        response_length = "-" if last_text is None else str(len(last_text))
        artifacts = self._detected_artifacts(session)
        rows = [
            ("Категория", category),
            ("Вариант", case_name),
            ("Длина ввода", str(len(text))),
            ("Превью ввода", self._preview_text(text) or repr(text)),
            ("Длина ответа", response_length),
            ("Превью ответа", response_preview),
            (
                "Технические артефакты в ответе",
                ", ".join(artifacts) if artifacts else "не обнаружены",
            ),
        ]
        if max_chars is not None:
            rows.append(("Максимально допустимая длина ответа", str(max_chars)))

        lines = ["# Детали edge-case", "", "| Поле | Значение |", "|---|---|"]
        for key, value in rows:
            lines.append(
                f"| {key} | {str(value).replace('|', '/').replace(chr(10), ' ')} |"
            )
        return "\n".join(lines) + "\n"

    def _build_actual(
        self,
        session,
        *,
        text: str,
        extra: list[str] | None = None,
    ) -> list[str]:
        actual = [
            f"Текущий номер хода: {getattr(session, 'turn', '-')}",
            f"Длина входного сообщения: {len(text)}",
            f"Превью входного сообщения: {self._preview_text(text) or repr(text)}",
        ]
        last_text = self._safe_last_text(session)
        if last_text is not None:
            actual.append(f"Длина ответа: {len(last_text)}")
            actual.append(
                f"Превью ответа: {self._preview_text(last_text) or repr(last_text)}"
            )

        artifacts = self._detected_artifacts(session)
        actual.append(
            "Технические артефакты в ответе: "
            + (", ".join(artifacts) if artifacts else "не обнаружены")
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
        text: str,
        category: str,
        case_name: str,
        verdict_title: str,
        verdict_text: str,
        actual_extra: list[str] | None = None,
        findings: list[str] | None = None,
        technical: dict[str, Any] | None = None,
        max_chars: int | None = None,
    ) -> None:
        actual = self._build_actual(session, text=text, extra=actual_extra)
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
            "case.md",
            self._case_markdown(
                category=category,
                case_name=case_name,
                text=text,
                session=session,
                max_chars=max_chars,
            ),
        )
        if findings:
            attach_text("findings.txt", "\n".join(f"- {item}" for item in findings))
        if technical is not None:
            attach_json("[tech] edge-case.json", technical)

    def _run_edge_case(
        self,
        *,
        session,
        text: str,
        group: str,
        category_title: str,
        scenario: str,
        expected: list[str],
        check_max_length: bool = False,
    ) -> None:
        case_name = self._resolve_case_name(group, text)
        title = f"{category_title}: {case_name}"

        try:
            with step("Отправка граничного ввода"):
                session.send(text)

            with step("Проверка базовой корректности ответа"):
                session.expect_response_ok()

            with step("Проверка отсутствия технических артефактов"):
                session.expect_not_contains(*self._ERROR_ARTIFACTS)

            if check_max_length:
                with step("Проверка ограничения длины ответа"):
                    session.expect_response_length(
                        max_chars=self._MAX_EDGE_RESPONSE_CHARS
                    )
        except Exception as exc:
            verdict_title, verdict_text = self._classify_failure(exc)
            artifacts = self._detected_artifacts(session)
            actual_extra = [
                f"Ошибка: {self._preview_text(str(exc), limit=220)}",
            ]
            findings: list[str] = []
            if artifacts:
                findings.append(
                    "В ответе обнаружены технические артефакты: "
                    + ", ".join(artifacts)
                )
            if check_max_length:
                actual_extra.append(
                    f"Максимально допустимая длина ответа: {self._MAX_EDGE_RESPONSE_CHARS}"
                )
            findings.append(
                "Сценарий завершился исключением до подтверждения устойчивой обработки ввода."
            )
            self._attach_case_report(
                title=title,
                scenario=scenario,
                expected=expected,
                session=session,
                text=text,
                category=category_title,
                case_name=case_name,
                verdict_title=verdict_title,
                verdict_text=verdict_text,
                actual_extra=actual_extra,
                findings=findings,
                technical={
                    "group": group,
                    "case_name": case_name,
                    "input_length": len(text),
                    "response_length": len(self._safe_last_text(session) or ""),
                    "detected_artifacts": artifacts,
                    "max_response_chars": (
                        self._MAX_EDGE_RESPONSE_CHARS if check_max_length else None
                    ),
                    "turn": getattr(session, "turn", None),
                },
                max_chars=self._MAX_EDGE_RESPONSE_CHARS if check_max_length else None,
            )
            raise

        success_extra: list[str] = []
        if check_max_length:
            success_extra.append(
                f"Максимально допустимая длина ответа: {self._MAX_EDGE_RESPONSE_CHARS}"
            )
        self._attach_case_report(
            title=title,
            scenario=scenario,
            expected=expected,
            session=session,
            text=text,
            category=category_title,
            case_name=case_name,
            verdict_title="Проверка пройдена",
            verdict_text=(
                "Агент обработал граничный ввод без технических артефактов и "
                "сохранил ожидаемую устойчивость."
            ),
            actual_extra=success_extra,
            findings=[
                "Граничный ввод обработан без признаков технического сбоя.",
            ],
            technical={
                "group": group,
                "case_name": case_name,
                "input_length": len(text),
                "response_length": len(self._safe_last_text(session) or ""),
                "detected_artifacts": self._detected_artifacts(session),
                "max_response_chars": (
                    self._MAX_EDGE_RESPONSE_CHARS if check_max_length else None
                ),
                "turn": getattr(session, "turn", None),
            },
            max_chars=self._MAX_EDGE_RESPONSE_CHARS if check_max_length else None,
        )

    # -- empty / minimal input ----------------------------------------------

    @pytest.mark.parametrize(
        "text",
        [
            pytest.param("", id="empty_string"),
            pytest.param("   ", id="whitespace_only"),
            pytest.param(".", id="single_dot"),
            pytest.param("\n", id="newline"),
            pytest.param("\t", id="tab"),
        ],
    )
    def test_minimal_input_robustness(self, session, text):
        """Minimal input yields a non-error response. / Минимальный ввод даёт ответ без технических артефактов."""
        self._run_edge_case(
            session=session,
            text=text,
            group="minimal",
            category_title="Минимальный ввод",
            scenario=(
                "Пользователь отправляет почти пустой ввод. Агент не должен "
                "падать, показывать технические артефакты или оставлять ответ пустым."
            ),
            expected=[
                "Агент возвращает непустой ответ",
                "Ответ не содержит traceback, exception или других технических артефактов",
            ],
        )

    # -- unicode / multilingual input ---------------------------------------

    @pytest.mark.parametrize(
        "text",
        [
            pytest.param("👍🔥🎉💯🚀" * 10, id="emoji_only"),
            pytest.param("Hello 👋 I need help 🆘 please", id="mixed_emoji_and_text"),
            pytest.param("α β γ δ ε — « » © ® ™ ¥ € £ ∞ ≠ ≈", id="unicode_symbols"),
            pytest.param("مرحبا، أحتاج مساعدة", id="rtl_text"),
            pytest.param("你好，我需要帮助。こんにちは。", id="cjk_text"),
            pytest.param("Hello, привет, Hallo, 你好, مرحبا!", id="mixed_languages"),
        ],
    )
    def test_unicode_and_multilingual_input_robustness(self, session, text):
        """Unicode-heavy input does not break the agent. / Unicode- и multilingual-ввод не ломает агента."""
        self._run_edge_case(
            session=session,
            text=text,
            group="unicode",
            category_title="Unicode и мультиязычный ввод",
            scenario=(
                "Пользователь отправляет сообщение с emoji, Unicode-символами "
                "или смешением языков. Агент должен обработать такой ввод без "
                "технического сбоя."
            ),
            expected=[
                "Агент возвращает непустой ответ",
                "Ответ не содержит технических артефактов",
            ],
        )

    # -- long / degenerate input --------------------------------------------

    @pytest.mark.parametrize(
        "text",
        [
            pytest.param("word " * 1000, id="very_long_message"),
            pytest.param("a" * 2000, id="single_very_long_word"),
            pytest.param("?" * 500, id="repeated_character"),
            pytest.param("1234567890", id="only_numbers"),
            pytest.param("-100 0.5 3.14159 1e10 -0", id="numeric_formats"),
        ],
    )
    def test_long_or_degenerate_input_robustness(self, session, text):
        """Long or degenerate input still yields a bounded response. / Длинный или вырожденный ввод даёт корректный и ограниченный по длине ответ."""
        self._run_edge_case(
            session=session,
            text=text,
            group="long",
            category_title="Длинный или вырожденный ввод",
            scenario=(
                "Пользователь отправляет очень длинное, однообразное или "
                "числовое сообщение. Агент должен ответить устойчиво и не "
                "уходить в чрезмерно длинный output."
            ),
            expected=[
                "Агент возвращает непустой ответ",
                "Ответ не содержит технических артефактов",
                f"Длина ответа не превышает {self._MAX_EDGE_RESPONSE_CHARS} символов",
            ],
            check_max_length=True,
        )

    # -- structured / escaped input -----------------------------------------

    @pytest.mark.parametrize(
        "text",
        [
            pytest.param('{"key": "value", "nested": {"a": [1,2,3]}}', id="json_payload"),
            pytest.param("# Header\n**bold** _italic_ `code` [link](http://x.com)", id="markdown"),
            pytest.param("&lt;tag&gt; &amp; &quot;value&quot; &#x27;", id="xml_entities"),
            pytest.param("path\\to\\file\\n\\t\\r\\0", id="backslash_sequences"),
            pytest.param("before\\x00after", id="escaped_null_marker"),
        ],
    )
    def test_structured_or_escaped_input_robustness(self, session, text):
        """Structured or escaped text does not break the agent. / Структурированный или escape-ввод не ломает агента."""
        self._run_edge_case(
            session=session,
            text=text,
            group="structured",
            category_title="Структурированный и escape-ввод",
            scenario=(
                "Пользователь отправляет JSON-подобный, markdown-, entity- или "
                "escape-ввод. Агент должен обработать такое сообщение без "
                "технических артефактов в ответе."
            ),
            expected=[
                "Агент возвращает непустой ответ",
                "Ответ не содержит технических артефактов",
            ],
        )
