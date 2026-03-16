"""
Generic edge-case tests applicable to any LLM agent.

Requires a ``session`` fixture returning an initialized
:class:`~agent_test_kit.AgentSession`.

Универсальные тесты граничных случаев, применимые к любому LLM-агенту.
Требуется фикстура ``session``, возвращающая инициализированный
:class:`~agent_test_kit.AgentSession`.
"""
from __future__ import annotations

import pytest


class GenericEdgeCaseTests:
    """Edge-case inputs that should never crash the agent.

    Входные данные граничных случаев, которые не должны приводить к сбою агента.
    """

    # -- empty / minimal input ----------------------------------------------

    @pytest.mark.parametrize("text", [
        pytest.param("", id="empty_string"),
        pytest.param("   ", id="whitespace_only"),
        pytest.param(".", id="single_dot"),
        pytest.param("\n", id="newline"),
        pytest.param("\t", id="tab"),
    ])
    def test_minimal_input(self, session, text):
        """Agent responds without error to empty / minimal input. / Агент отвечает без ошибки на пустой или минимальный ввод."""
        session.send(text)
        session.expect_response_ok()

    # -- emoji / unicode ----------------------------------------------------

    def test_emoji_only(self, session):
        """Emoji-only input does not crash the agent. / Ввод только эмодзи не приводит к сбою агента."""
        session.send("👍🔥🎉💯🚀" * 10)
        session.expect_response_ok()

    def test_mixed_emoji_and_text(self, session):
        """Agent handles a message mixing emoji and text. / Агент обрабатывает сообщение с эмодзи и текстом."""
        session.send("Hello 👋 I need help 🆘 please")
        session.expect_response_ok()

    def test_unicode_special_chars(self, session):
        """Unicode symbols do not break the agent. / Символы Unicode не ломают агента."""
        session.send("α β γ δ ε — « » © ® ™ ¥ € £ ∞ ≠ ≈")
        session.expect_response_ok()

    def test_rtl_text(self, session):
        """Right-to-left text does not break the agent. / Текст справа налево не ломает агента."""
        session.send("مرحبا، أحتاج مساعدة")
        session.expect_response_ok()

    def test_cjk_characters(self, session):
        """CJK characters do not break the agent. / Символы CJK не ломают агента."""
        session.send("你好，我需要帮助。こんにちは。")
        session.expect_response_ok()

    # -- long messages ------------------------------------------------------

    def test_very_long_message(self, session):
        """Agent handles a message of ~5000 characters. / Агент обрабатывает сообщение ~5000 символов."""
        session.send("word " * 1000)
        session.expect_response_ok()

    def test_single_very_long_word(self, session):
        """A single ~2000-character word does not crash the agent. / Одно слово ~2000 символов не приводит к сбою агента."""
        session.send("a" * 2000)
        session.expect_response_ok()

    # -- special characters / injection attempts ----------------------------

    def test_html_tags(self, session):
        """HTML tags in input do not break the agent or appear in output. / HTML-теги во вводе не ломают агента и не появляются в выводе."""
        session.send("<script>alert('xss')</script><img src=x onerror=alert(1)>")
        session.expect_response_ok()
        session.expect_not_contains("<script>", "onerror")

    def test_json_in_input(self, session):
        """JSON payload in user message does not break the agent. / JSON-данные в сообщении пользователя не ломают агента."""
        session.send('{"key": "value", "nested": {"a": [1,2,3]}}')
        session.expect_response_ok()

    def test_markdown_formatting(self, session):
        """Markdown syntax in input does not break the agent. / Синтаксис Markdown во вводе не ломает агента."""
        session.send("# Header\n**bold** _italic_ `code` [link](http://x.com)")
        session.expect_response_ok()

    def test_sql_like_input(self, session):
        """SQL-like fragments in input do not break the agent. / SQL-подобные фрагменты во вводе не ломают агента."""
        session.send("SELECT * FROM users WHERE id = 1; DROP TABLE users;--")
        session.expect_response_ok()

    def test_xml_entities(self, session):
        """XML entities in input do not break the agent. / XML-сущности во вводе не ломают агента."""
        session.send("&lt;tag&gt; &amp; &quot;value&quot; &#x27;")
        session.expect_response_ok()

    def test_backslash_sequences(self, session):
        """Backslash escape sequences do not break the agent. / Escape-последовательности с обратным слэшем не ломают агента."""
        session.send("path\\to\\file\\n\\t\\r\\0")
        session.expect_response_ok()

    def test_null_bytes(self, session):
        """Null-byte-like patterns do not break the agent. / Паттерны с нулевым байтом не ломают агента."""
        session.send("before\x00after")
        session.expect_response_ok()

    # -- repeated / duplicate input -----------------------------------------

    def test_repeated_character(self, session):
        """Repeating a single character ~500 times does not crash the agent. / Повтор символа ~500 раз не приводит к сбою агента."""
        session.send("?" * 500)
        session.expect_response_ok()

    def test_same_message_twice(self, session):
        """Sending the same message twice does not confuse the agent. / Отправка одного и того же сообщения дважды не сбивает агента с толку."""
        session.send("Hello")
        session.expect_response_ok()
        session.send("Hello")
        session.expect_response_ok()

    # -- numbers & edge formats ---------------------------------------------

    def test_only_numbers(self, session):
        """A message containing only numbers is handled. / Сообщение только с числами обрабатывается."""
        session.send("1234567890")
        session.expect_response_ok()

    def test_negative_and_decimal_numbers(self, session):
        """Negative and decimal numbers do not confuse the agent. / Отрицательные и десятичные числа не сбивают агента с толку."""
        session.send("-100 0.5 3.14159 1e10 -0")
        session.expect_response_ok()

    # -- mixed language -----------------------------------------------------

    def test_mixed_languages(self, session):
        """Input mixing multiple languages is handled. / Ввод на нескольких языках обрабатывается."""
        session.send("Hello, привет, Hallo, 你好, مرحبا!")
        session.expect_response_ok()
