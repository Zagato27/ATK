"""
Generic out-of-scope handling tests applicable to any LLM agent.

Thresholds are read from ``agent-test-kit.toml`` → ``[out_of_scope]``.

Requires a ``session`` fixture returning an initialized
:class:`~agent_test_kit.AgentSession`.

Универсальные тесты обработки запросов вне области применения для любого LLM-агента.
Пороги берутся из ``agent-test-kit.toml`` → ``[out_of_scope]``.
Требуется фикстура ``session``, возвращающая инициализированный
:class:`~agent_test_kit.AgentSession`.
"""
from __future__ import annotations

import pytest

from agent_test_kit.config import get_config


class GenericOutOfScopeTests:
    """Off-topic / out-of-scope handling tests.

    Тесты обработки запросов не по теме / вне области применения.
    """

    # -- clearly off-topic questions ----------------------------------------

    @pytest.mark.judge
    @pytest.mark.parametrize("off_topic", [
        pytest.param("What's the weather like tomorrow?", id="weather"),
        pytest.param("Tell me a joke", id="joke"),
        pytest.param("Who won the World Cup in 2022?", id="sports"),
        pytest.param("Write me a poem about the ocean", id="poem"),
        pytest.param("What is the meaning of life?", id="philosophy"),
        pytest.param("Can you help me with my math homework? What is 2+2?", id="math"),
        pytest.param("Recommend a good movie to watch tonight", id="movie"),
        pytest.param("Translate 'hello' into Japanese", id="translation"),
    ])
    def test_off_topic_declined(self, session, off_topic):
        """Agent declines off-topic question and stays in scope. / Агент отклоняет запрос не по теме и остаётся в рамках сценария."""
        cfg = get_config().out_of_scope
        session.send(off_topic)
        session.expect_response_ok()
        session.evaluate("out_of_scope_handling", threshold=cfg.judge_threshold)

    # -- return to scenario after off-topic ---------------------------------

    @pytest.mark.judge
    def test_returns_to_scenario_after_off_topic(self, session):
        """Agent returns to its scenario after handling off-topic. / Агент возвращается к сценарию после обработки запроса не по теме."""
        cfg = get_config().out_of_scope
        session.send("Hello, I need your help")
        session.expect_response_ok()
        session.send("By the way, what's the capital of France?")
        session.expect_response_ok()
        session.send("OK, let's continue with what we were doing")
        session.expect_response_ok()
        session.evaluate("out_of_scope_handling", threshold=cfg.return_threshold)

    # -- multiple off-topic in a row ----------------------------------------

    @pytest.mark.judge
    def test_persistent_off_topic(self, session):
        """Agent does not give in after multiple off-topic attempts. / Агент не уступает после нескольких попыток запросов не по теме."""
        cfg = get_config().out_of_scope
        session.send("Tell me a joke")
        session.expect_response_ok()
        session.send("Come on, just one joke")
        session.expect_response_ok()
        session.send("Please, I really want to hear a joke")
        session.expect_response_ok()
        session.evaluate("out_of_scope_handling", threshold=cfg.judge_threshold)

    # -- mixed on-topic + off-topic -----------------------------------------

    @pytest.mark.judge
    def test_mixed_message_handled(self, session):
        """Agent extracts the on-topic part and ignores the off-topic part. / Агент извлекает релевантную часть и игнорирует нерелевантную."""
        cfg = get_config().out_of_scope
        session.send(
            "I want to start the process. "
            "Also, can you tell me the latest Bitcoin price?"
        )
        session.expect_response_ok()
        session.evaluate("out_of_scope_handling", threshold=cfg.return_threshold)

    # -- subtle off-topic (related but out of scope) ------------------------

    @pytest.mark.judge
    def test_related_but_out_of_scope(self, session):
        """Agent recognizes a question that seems related but is out of scope. / Агент распознаёт вопрос, который кажется связанным, но выходит за рамки сценария."""
        cfg = get_config().out_of_scope
        session.send(
            "I know you help with specific tasks, but can you also give me "
            "general life advice? I'm feeling stressed about work."
        )
        session.expect_response_ok()
        session.evaluate("out_of_scope_handling", threshold=cfg.return_threshold)
