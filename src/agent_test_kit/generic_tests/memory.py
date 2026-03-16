"""
Generic memory & context retention tests applicable to any LLM agent.

Thresholds are read from ``agent-test-kit.toml`` → ``[memory]``.

Requires a ``session`` fixture returning an initialized
:class:`~agent_test_kit.AgentSession`.

Универсальные тесты памяти и удержания контекста для любого LLM-агента.
Пороги берутся из ``agent-test-kit.toml`` → ``[memory]``.
Требуется фикстура ``session``, возвращающая инициализированный
:class:`~agent_test_kit.AgentSession`.
"""
from __future__ import annotations

import pytest

from agent_test_kit.config import get_config
from agent_test_kit.helpers import run_dialogue


class GenericMemoryTests:
    """Memory and context retention tests.

    Тесты памяти и удержания контекста.
    """

    # -- basic context retention --------------------------------------------

    def test_remembers_user_name(self, session):
        """Agent remembers user's name across turns. / Агент запоминает имя пользователя между репликами."""
        session.send("My name is Alexander")
        session.expect_response_ok()
        session.send("What should we do next?")
        session.expect_response_ok()
        session.send("Do you remember my name?")
        session.expect_response_ok()
        session.expect_contains_any("alexander", "Alexander")

    @pytest.mark.judge
    def test_remembers_stated_preference(self, session):
        """Agent remembers a stated preference. / Агент запоминает заявленное предпочтение пользователя."""
        cfg = get_config().memory
        session.send("I prefer email communication")
        session.expect_response_ok()
        run_dialogue(session, ["Continue"] * 3)
        session.send("How did I say I prefer to be contacted?")
        session.expect_response_ok()
        session.evaluate("context_retention", threshold=cfg.context_threshold)

    # -- no repeated questions ----------------------------------------------

    @pytest.mark.judge
    def test_no_repeated_question_after_answer(self, session):
        """Agent does not re-ask a question that was already answered. / Агент не переспрашивает уже отвеченный вопрос."""
        session.send("My name is Alexander")
        session.expect_response_ok()
        session.send("Let's continue")
        session.expect_response_ok()
        # Check for re-asking name / Проверка повторного запроса имени
        second_response = session.last_text.lower()
        asks_name_again = any(
            phrase in second_response
            for phrase in ["your name", "ваше имя", "как вас зовут", "what is your name"]
        )
        assert not asks_name_again, (
            f"Agent re-asked for name in: {session.last_text[:300]}"
        )

    # -- correction handling ------------------------------------------------

    @pytest.mark.judge
    def test_accepts_correction(self, session):
        """Agent updates data when user corrects themselves. / Агент обновляет данные при исправлении пользователем."""
        cfg = get_config().memory
        session.send("My age is 30")
        session.expect_response_ok()
        session.send("Sorry, I made a mistake — I'm actually 31")
        session.expect_response_ok()
        session.evaluate("correction_handling", threshold=cfg.correction_threshold)

    @pytest.mark.judge
    def test_correction_persists(self, session):
        """Corrected data is used in subsequent turns. / Исправленные данные используются в последующих репликах."""
        cfg = get_config().memory
        session.send("I live in London")
        session.expect_response_ok()
        session.send("Actually, I moved — I live in Berlin now")
        session.expect_response_ok()
        session.send("Can you confirm where I live?")
        session.expect_response_ok()
        session.evaluate("context_retention", threshold=cfg.context_threshold)

    # -- context across many turns ------------------------------------------

    @pytest.mark.judge
    @pytest.mark.slow
    def test_context_after_10_turns(self, session):
        """Agent retains context after 10 turns of conversation. / Агент сохраняет контекст после 10 реплик диалога."""
        cfg = get_config().memory
        session.send("Important: my reference number is REF-98765")
        session.expect_response_ok()
        run_dialogue(session, ["Tell me more"] * 9)
        session.send("What was my reference number?")
        session.expect_response_ok()
        session.evaluate("context_retention", threshold=cfg.context_threshold)

    @pytest.mark.judge
    @pytest.mark.slow
    def test_context_after_20_turns(self, session):
        """Agent retains context after 20 turns of conversation. / Агент сохраняет контекст после 20 реплик диалога."""
        cfg = get_config().memory
        session.send("Remember this code: ALPHA-2025")
        session.expect_response_ok()
        run_dialogue(session, ["Continue please"] * 18)
        session.send("What code did I ask you to remember?")
        session.expect_response_ok()
        session.evaluate("context_retention", threshold=cfg.long_context_threshold)

    # -- multiple facts retained simultaneously -----------------------------

    @pytest.mark.judge
    def test_multiple_facts_retained(self, session):
        """Agent retains multiple distinct facts provided in separate turns. / Агент сохраняет несколько фактов, сообщённых в разных репликах."""
        cfg = get_config().memory
        session.send("My name is Maria")
        session.expect_response_ok()
        session.send("I am 28 years old")
        session.expect_response_ok()
        session.send("I work as an engineer")
        session.expect_response_ok()
        session.send(
            "Can you summarize what you know about me?"
        )
        session.expect_response_ok()
        session.evaluate("context_retention", threshold=cfg.context_threshold)

    # -- context after off-topic interruption --------------------------------

    @pytest.mark.judge
    def test_context_survives_off_topic(self, session):
        """Agent retains context even after an off-topic detour. / Агент сохраняет контекст даже после отвлечения на запрос не по теме."""
        cfg = get_config().memory
        session.send("My account ID is ACC-555")
        session.expect_response_ok()
        session.send("By the way, what's the weather like?")
        session.expect_response_ok()
        session.send("OK, back to business. What's my account ID?")
        session.expect_response_ok()
        session.evaluate("context_retention", threshold=cfg.context_threshold)
