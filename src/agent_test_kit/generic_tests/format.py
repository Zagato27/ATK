"""
Generic communication format tests applicable to any LLM agent.

Thresholds and forbidden-token lists are read from
``agent-test-kit.toml`` → ``[format]``.

Requires a ``session`` fixture returning an initialized
:class:`~agent_test_kit.AgentSession`.

Универсальные тесты формата коммуникации, применимые к любому LLM-агенту.
Пороговые значения и списки запрещённых токенов читаются из
``agent-test-kit.toml`` → ``[format]``. Требуется фикстура ``session``,
возвращающая инициализированный :class:`~agent_test_kit.AgentSession`.
"""
from __future__ import annotations

import pytest

from agent_test_kit.config import get_config


class GenericFormatTests:
    """Format & style assertions for any LLM agent.

    Проверки формата и стиля для любого LLM-агента.
    """

    # -- basic response structure -------------------------------------------

    def test_response_not_empty(self, session):
        """Response to a simple greeting is non-empty. / Ответ на простое приветствие не пустой."""
        cfg = get_config().format
        session.send("Hello")
        session.expect_response_ok()
        session.expect_response_length(min_chars=cfg.min_response_length)

    def test_response_not_excessively_long(self, session):
        """A simple question does not produce excessively long output. / Простой вопрос не даёт чрезмерно длинного вывода."""
        cfg = get_config().format
        session.send("What can you help me with?")
        session.expect_response_ok()
        session.expect_response_length(max_chars=cfg.max_response_length)

    def test_response_to_short_question(self, session):
        """Short question gets a reasonable-length response. / Короткий вопрос получает ответ разумной длины."""
        cfg = get_config().format
        session.send("Help")
        session.expect_response_ok()
        session.expect_response_length(
            min_chars=cfg.short_response_min,
            max_chars=cfg.short_response_max,
        )

    # -- encoding & rendering -----------------------------------------------

    def test_no_raw_html_in_response(self, session):
        """Agent response does not contain raw HTML tags. / Ответ агента не содержит сырых HTML-тегов."""
        cfg = get_config().format
        session.send("Hello, how are you?")
        session.expect_response_ok()
        session.expect_not_contains(*cfg.forbidden_html)

    def test_no_internal_tokens(self, session):
        """Agent does not leak internal tokens or markers. / Агент не раскрывает внутренние токены или маркеры."""
        cfg = get_config().format
        session.send("Tell me about yourself")
        session.expect_response_ok()
        session.expect_not_contains(*cfg.forbidden_tokens)

    def test_no_placeholder_text(self, session):
        """Agent does not output placeholder / template strings. / Агент не выводит плейсхолдеры или шаблонные строки."""
        cfg = get_config().format
        session.send("Hello")
        session.expect_response_ok()
        session.expect_not_contains(*cfg.forbidden_placeholders)

    # -- conversational flow ------------------------------------------------

    @pytest.mark.judge
    def test_greeting_politeness(self, session):
        """Agent responds to greeting politely. / Агент вежливо отвечает на приветствие."""
        threshold = get_config().evaluate.default_threshold
        session.send("Hello!")
        session.expect_response_ok()
        session.evaluate("politeness", threshold=threshold)

    @pytest.mark.judge
    def test_polite_under_rude_input(self, session):
        """Agent stays polite even when the user is rude. / Агент остаётся вежливым даже при грубом вводе пользователя."""
        threshold = get_config().evaluate.default_threshold
        session.send("This is terrible, you are useless, just do your job!")
        session.expect_response_ok()
        session.evaluate("politeness", threshold=threshold)

    def test_first_response_asks_or_guides(self, session):
        """First response to a vague message asks a question or offers guidance. / Первый ответ на неясное сообщение задаёт вопрос или предлагает помощь."""
        session.send("I need help")
        session.expect_response_ok()
        text = session.last_text
        has_question = "?" in text
        has_guidance = any(w in text.lower() for w in [
            "can", "would", "please", "let me", "how",
            "могу", "можете", "пожалуйста", "давайте", "как",
        ])
        assert has_question or has_guidance, (
            f"Expected a question or guidance, got: {text[:300]}"
        )

    # -- no error artifacts -------------------------------------------------

    def test_no_error_messages(self, session):
        """Agent does not expose raw error messages to the user. / Агент не раскрывает сырые сообщения об ошибках пользователю."""
        cfg = get_config().format
        session.send("Hello, let's start")
        session.expect_response_ok()
        session.expect_not_contains(*cfg.forbidden_errors)

    def test_no_debug_artifacts(self, session):
        """Agent response does not contain debug / log artifacts. / Ответ агента не содержит артефактов отладки или логов."""
        cfg = get_config().format
        session.send("Hi")
        session.expect_response_ok()
        session.expect_not_contains(*cfg.forbidden_debug)

    # -- consistency across turns -------------------------------------------

    def test_consistent_language(self, session):
        """Agent responds in the same language the user wrote in. Override this test if the agent always replies in a fixed language. / Агент отвечает на том же языке, на котором писал пользователь. Переопределите этот тест, если агент всегда отвечает на фиксированном языке."""
        session.send("Hello, can you help me?")
        session.expect_response_ok()
        first_text = session.last_text

        session.send("What can you do?")
        session.expect_response_ok()
        second_text = session.last_text

        def _has_latin(t: str) -> bool:
            return any("a" <= c <= "z" or "A" <= c <= "Z" for c in t)

        assert _has_latin(first_text) == _has_latin(second_text), (
            "Language inconsistency between turns"
        )
