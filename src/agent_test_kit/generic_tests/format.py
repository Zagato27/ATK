"""
Generic surface-format tests applicable to any LLM agent.

Thresholds and forbidden-fragment lists are read from
``agent-test-kit.toml`` → ``[format]``.

Requires a ``session`` fixture returning an initialized
:class:`~agent_test_kit.AgentSession`.

Универсальные тесты поверхностного формата ответа, применимые к любому
LLM-агенту. Пороговые значения и списки запрещённых фрагментов читаются из
``agent-test-kit.toml`` → ``[format]``. Требуется фикстура ``session``,
возвращающая инициализированный :class:`~agent_test_kit.AgentSession`.
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


class GenericSurfaceFormatTests:
    """Surface-format and rendering-hygiene assertions for any LLM agent.

    Проверки поверхностного формата и артефактов рендера для любого LLM-агента.
    """

    _BASIC_CASES = {
        "Hello": "приветствие",
        "Help": "короткий запрос помощи",
        "What can you help me with?": "вопрос о возможностях",
    }
    _ARTIFACT_CASES = {
        "forbidden_html": "без фрагментов разметки",
        "forbidden_tokens": "без внутренних токенов",
        "forbidden_placeholders": "без placeholder-текста",
        "forbidden_errors": "без error-артефактов",
        "forbidden_debug": "без debug-артефактов",
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

        if self._looks_like_environment_error(message):
            return (
                "Проблема окружения или доступности сервиса",
                "Проверка не дошла до оценки формата ответа, потому что запрос "
                "завершился сетевой или серверной ошибкой.",
            )
        if "response text is empty" in lower:
            return (
                "Пустой ответ",
                "Агент формально ответил, но не вернул пользовательского текста.",
            )
        if lower.startswith("turn") and "http " in lower:
            return (
                "Сервис вернул ошибку",
                "Формат ответа нельзя оценить, потому что сервис не вернул успешный HTTP-ответ.",
            )
        if case == "shape" and "length " in lower and "expected [" in lower:
            return (
                "Ответ не укладывается в ожидаемые границы длины",
                "Пользовательский ответ оказался слишком коротким или слишком длинным для данного базового сценария.",
            )
        if case == "bytes" and "byte limit" in lower:
            return (
                "Превышен бюджет по байтам",
                "Ответ содержит слишком много данных и выходит за допустимый размер.",
            )
        if case == "artifacts" and "contains forbidden" in lower:
            return (
                "В ответе видны артефакты рендера или реализации",
                "Пользователь увидел служебные токены, debug-фрагменты, placeholder или другие поверхностные артефакты.",
            )
        return (
            "Нарушено ожидаемое поведение",
            "Результат сценария не соответствует ожидаемому пользовательскому формату ответа.",
        )

    def _detected_forbidden(self, session, forbidden: list[str]) -> list[str]:
        last_text = self._safe_last_text(session)
        if not last_text:
            return []
        lower = last_text.lower()
        return [item for item in forbidden if item.lower() in lower]

    def _build_actual(
        self,
        session,
        *,
        prompt: str,
        extra: list[str] | None = None,
    ) -> list[str]:
        actual = [
            f"Текущий номер хода: {getattr(session, 'turn', '-')}",
            f"Превью запроса: {self._preview_text(prompt) or repr(prompt)}",
        ]
        last_text = self._safe_last_text(session)
        if last_text is not None:
            actual.append(f"Длина ответа в символах: {len(last_text)}")
            actual.append(f"Размер ответа в байтах: {len(last_text.encode('utf-8'))}")
            actual.append(
                f"Превью ответа: {self._preview_text(last_text) or repr(last_text)}"
            )
        if extra:
            actual.extend(extra)
        return actual

    def _response_markdown(
        self,
        *,
        category: str,
        case_name: str,
        prompt: str,
        session,
        limits: list[tuple[str, str]],
    ) -> str:
        last_text = self._safe_last_text(session)
        rows = [
            ("Категория", category),
            ("Вариант", case_name),
            ("Превью запроса", self._preview_text(prompt) or repr(prompt)),
            ("Длина запроса", str(len(prompt))),
            ("Длина ответа", "-" if last_text is None else str(len(last_text))),
            (
                "Размер ответа в байтах",
                "-" if last_text is None else str(len(last_text.encode("utf-8"))),
            ),
            (
                "Превью ответа",
                self._preview_text(last_text) or "-",
            ),
        ]
        rows.extend(limits)
        lines = ["# Детали format-case", "", "| Поле | Значение |", "|---|---|"]
        for key, value in rows:
            lines.append(
                f"| {key} | {str(value).replace('|', '/').replace(chr(10), ' ')} |"
            )
        return "\n".join(lines) + "\n"

    def _attach_case_report(
        self,
        *,
        title: str,
        scenario: str,
        expected: list[str],
        session,
        prompt: str,
        category: str,
        case_name: str,
        verdict_title: str,
        verdict_text: str,
        actual_extra: list[str] | None = None,
        findings: list[str] | None = None,
        limits: list[tuple[str, str]] | None = None,
        technical: dict[str, Any] | None = None,
    ) -> None:
        actual = self._build_actual(session, prompt=prompt, extra=actual_extra)
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
            "response.md",
            self._response_markdown(
                category=category,
                case_name=case_name,
                prompt=prompt,
                session=session,
                limits=limits or [],
            ),
        )
        if findings:
            attach_text("findings.txt", "\n".join(f"- {item}" for item in findings))
        if technical is not None:
            attach_json("[tech] format-case.json", technical)

    # -- basic response shape ------------------------------------------------

    @pytest.mark.parametrize(
        ("prompt", "min_attr", "max_attr"),
        [
            pytest.param("Hello", "min_response_length", "max_response_length", id="greeting"),
            pytest.param("Help", "short_response_min", "short_response_max", id="short_help"),
            pytest.param(
                "What can you help me with?",
                "min_response_length",
                "max_response_length",
                id="capabilities",
            ),
        ],
    )
    def test_basic_response_shape(self, session, prompt, min_attr, max_attr):
        """Basic prompts receive bounded user-facing text. / Базовые запросы получают пользовательский текст в разумных пределах."""
        cfg = get_config().format
        min_chars = getattr(cfg, min_attr)
        max_chars = getattr(cfg, max_attr)
        case_name = self._BASIC_CASES.get(prompt, "базовый формат ответа")
        title = f"Формат ответа: {case_name}"
        scenario = (
            "Пользователь задаёт базовый запрос. Агент должен вернуть "
            "понятный пользовательский ответ в ожидаемых границах длины."
        )
        expected = [
            "Агент возвращает непустой ответ",
            f"Длина ответа находится в диапазоне {min_chars}..{max_chars} символов",
        ]

        try:
            with step("Отправка базового запроса"):
                session.send(prompt)

            with step("Проверка базовой корректности ответа"):
                session.expect_response_ok()

            with step("Проверка границ длины ответа"):
                session.expect_response_length(
                    min_chars=min_chars,
                    max_chars=max_chars,
                )
        except Exception as exc:
            verdict_title, verdict_text = self._classify_failure(exc, case="shape")
            self._attach_case_report(
                title=title,
                scenario=scenario,
                expected=expected,
                session=session,
                prompt=prompt,
                category="Базовая форма ответа",
                case_name=case_name,
                verdict_title=verdict_title,
                verdict_text=verdict_text,
                actual_extra=[
                    f"Допустимый диапазон длины: {min_chars}..{max_chars}",
                    f"Ошибка: {self._preview_text(str(exc), limit=220)}",
                ],
                findings=[
                    "Базовый пользовательский ответ не уложился в ожидаемые границы длины или не прошёл базовую проверку."
                ],
                limits=[
                    ("Минимальная длина", str(min_chars)),
                    ("Максимальная длина", str(max_chars)),
                ],
                technical={
                    "case": "basic_response_shape",
                    "case_name": case_name,
                    "prompt": prompt,
                    "min_chars": min_chars,
                    "max_chars": max_chars,
                    "response_length": len(self._safe_last_text(session) or ""),
                    "turn": getattr(session, "turn", None),
                },
            )
            raise

        self._attach_case_report(
            title=title,
            scenario=scenario,
            expected=expected,
            session=session,
            prompt=prompt,
            category="Базовая форма ответа",
            case_name=case_name,
            verdict_title="Проверка пройдена",
            verdict_text=(
                "Ответ выглядит как нормальный пользовательский текст и "
                "укладывается в ожидаемые границы длины."
            ),
            actual_extra=[
                f"Допустимый диапазон длины: {min_chars}..{max_chars}",
            ],
            findings=[
                "Базовый ответ соответствует ожидаемой длине и не выглядит пустым.",
            ],
            limits=[
                ("Минимальная длина", str(min_chars)),
                ("Максимальная длина", str(max_chars)),
            ],
            technical={
                "case": "basic_response_shape",
                "case_name": case_name,
                "prompt": prompt,
                "min_chars": min_chars,
                "max_chars": max_chars,
                "response_length": len(self._safe_last_text(session) or ""),
                "turn": getattr(session, "turn", None),
            },
        )

    def test_response_size_within_byte_budget(self, session):
        """Responses stay within the configured byte budget. / Ответы укладываются в заданный бюджет по байтам."""
        cfg = get_config().format
        prompt = "Tell me everything you can help me with"
        title = "Размер ответа укладывается в лимит по байтам"
        scenario = (
            "Пользователь просит развёрнуто описать возможности агента. "
            "Даже подробный ответ должен оставаться в установленном бюджете по байтам."
        )
        expected = [
            "Агент возвращает непустой ответ",
            f"Размер ответа не превышает {cfg.max_response_bytes} байт",
        ]
        size = 0

        try:
            with step("Отправка запроса на развёрнутое описание возможностей"):
                session.send(prompt)

            with step("Проверка базовой корректности ответа"):
                session.expect_response_ok()

            with step("Проверка лимита по байтам"):
                size = len(session.last_text.encode("utf-8"))
                assert size <= cfg.max_response_bytes, (
                    f"Response size {size} bytes exceeds "
                    f"{cfg.max_response_bytes} byte limit"
                )
        except Exception as exc:
            verdict_title, verdict_text = self._classify_failure(exc, case="bytes")
            self._attach_case_report(
                title=title,
                scenario=scenario,
                expected=expected,
                session=session,
                prompt=prompt,
                category="Размер ответа",
                case_name="лимит по байтам",
                verdict_title=verdict_title,
                verdict_text=verdict_text,
                actual_extra=[
                    f"Лимит по байтам: {cfg.max_response_bytes}",
                    f"Фактический размер ответа: {size}",
                    f"Ошибка: {self._preview_text(str(exc), limit=220)}",
                ],
                findings=[
                    "Размер ответа превысил допустимый бюджет или ответ не прошёл базовую проверку."
                ],
                limits=[
                    ("Максимальный размер ответа, байт", str(cfg.max_response_bytes)),
                    ("Фактический размер ответа, байт", str(size)),
                ],
                technical={
                    "case": "response_size_within_byte_budget",
                    "prompt": prompt,
                    "max_response_bytes": cfg.max_response_bytes,
                    "actual_size_bytes": size,
                    "response_length": len(self._safe_last_text(session) or ""),
                    "turn": getattr(session, "turn", None),
                },
            )
            raise

        self._attach_case_report(
            title=title,
            scenario=scenario,
            expected=expected,
            session=session,
            prompt=prompt,
            category="Размер ответа",
            case_name="лимит по байтам",
            verdict_title="Проверка пройдена",
            verdict_text=(
                "Даже развёрнутый ответ остаётся в допустимом бюджете по байтам "
                "и не перегружает клиентскую сторону."
            ),
            actual_extra=[
                f"Лимит по байтам: {cfg.max_response_bytes}",
                f"Фактический размер ответа: {size}",
            ],
            findings=[
                "Размер ответа укладывается в установленное ограничение.",
            ],
            limits=[
                ("Максимальный размер ответа, байт", str(cfg.max_response_bytes)),
                ("Фактический размер ответа, байт", str(size)),
            ],
            technical={
                "case": "response_size_within_byte_budget",
                "prompt": prompt,
                "max_response_bytes": cfg.max_response_bytes,
                "actual_size_bytes": size,
                "response_length": len(self._safe_last_text(session) or ""),
                "turn": getattr(session, "turn", None),
            },
        )

    # -- rendering / artifact hygiene ---------------------------------------

    @pytest.mark.parametrize(
        ("prompt", "forbidden_attr"),
        [
            pytest.param("Hello, how are you?", "forbidden_html", id="no_markup_fragments"),
            pytest.param("Tell me about yourself", "forbidden_tokens", id="no_internal_tokens"),
            pytest.param("Hello", "forbidden_placeholders", id="no_placeholder_text"),
            pytest.param("Hello, let's start", "forbidden_errors", id="no_error_artifacts"),
            pytest.param("Hi", "forbidden_debug", id="no_debug_artifacts"),
        ],
    )
    def test_no_surface_artifacts(self, session, prompt, forbidden_attr):
        """Responses do not expose rendering or implementation artifacts. / Ответы не раскрывают артефакты рендера или реализации."""
        cfg = get_config().format
        forbidden = list(getattr(cfg, forbidden_attr))
        case_name = self._ARTIFACT_CASES.get(
            forbidden_attr,
            "без поверхностных артефактов",
        )
        title = f"Ответ не содержит артефактов: {case_name}"
        scenario = (
            "Пользователь отправляет обычный запрос. В ответе не должно быть "
            "служебных фрагментов, placeholder-текста, debug-следов или "
            "внутренних токенов."
        )
        expected = [
            "Агент возвращает непустой ответ",
            "Ответ не содержит запрещённых поверхностных артефактов",
        ]

        try:
            with step("Отправка пользовательского запроса"):
                session.send(prompt)

            with step("Проверка базовой корректности ответа"):
                session.expect_response_ok()

            with step("Проверка отсутствия запрещённых артефактов"):
                session.expect_not_contains(*forbidden)
        except Exception as exc:
            detected = self._detected_forbidden(session, forbidden)
            verdict_title, verdict_text = self._classify_failure(
                exc,
                case="artifacts",
            )
            self._attach_case_report(
                title=title,
                scenario=scenario,
                expected=expected,
                session=session,
                prompt=prompt,
                category="Артефакты рендера и реализации",
                case_name=case_name,
                verdict_title=verdict_title,
                verdict_text=verdict_text,
                actual_extra=[
                    f"Проверяемый список запрещённых фрагментов: {', '.join(forbidden)}",
                    (
                        "Найдены фрагменты: " + ", ".join(detected)
                        if detected
                        else "Найдены фрагменты: не удалось определить явно"
                    ),
                    f"Ошибка: {self._preview_text(str(exc), limit=220)}",
                ],
                findings=[
                    "В ответе появились пользовательски заметные поверхностные артефакты."
                    if detected
                    else "Проверка артефактов завершилась исключением до успешного подтверждения чистоты ответа.",
                ],
                limits=[
                    ("Количество запрещённых фрагментов", str(len(forbidden))),
                ],
                technical={
                    "case": "no_surface_artifacts",
                    "case_name": case_name,
                    "prompt": prompt,
                    "forbidden_attr": forbidden_attr,
                    "forbidden": forbidden,
                    "detected": detected,
                    "response_length": len(self._safe_last_text(session) or ""),
                    "turn": getattr(session, "turn", None),
                },
            )
            raise

        self._attach_case_report(
            title=title,
            scenario=scenario,
            expected=expected,
            session=session,
            prompt=prompt,
            category="Артефакты рендера и реализации",
            case_name=case_name,
            verdict_title="Проверка пройдена",
            verdict_text=(
                "Ответ выглядит чисто для пользователя и не раскрывает "
                "служебные или внутренние фрагменты."
            ),
            actual_extra=[
                f"Проверяемый список запрещённых фрагментов: {', '.join(forbidden)}",
                "Запрещённые поверхностные артефакты не обнаружены.",
            ],
            findings=[
                "Ответ не содержит заметных пользователю артефактов рендера или реализации.",
            ],
            limits=[
                ("Количество запрещённых фрагментов", str(len(forbidden))),
            ],
            technical={
                "case": "no_surface_artifacts",
                "case_name": case_name,
                "prompt": prompt,
                "forbidden_attr": forbidden_attr,
                "forbidden": forbidden,
                "detected": [],
                "response_length": len(self._safe_last_text(session) or ""),
                "turn": getattr(session, "turn", None),
            },
        )
