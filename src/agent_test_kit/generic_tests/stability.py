"""
Generic stability & consistency tests applicable to any LLM agent.

Thresholds are read from ``agent-test-kit.toml`` → ``[stability]``.

Requires fixtures:

- ``session`` — initialized :class:`~agent_test_kit.AgentSession`
- ``agent_client`` — a :class:`~agent_test_kit.BaseAgentClient` instance
  (``scope="session"``)
- ``judge_llm`` — (optional) a :class:`~agent_test_kit.BaseLLMJudge`

Универсальные тесты стабильности и согласованности для любого LLM-агента.
Пороги берутся из ``agent-test-kit.toml`` → ``[stability]``.
Требуются фикстуры: ``session``, ``agent_client``, ``judge_llm`` (опционально).
"""
from __future__ import annotations

import pytest

from agent_test_kit.config import get_config
from agent_test_kit.session import AgentSession


class GenericStabilityTests:
    """Stability & consistency tests.

    Тесты стабильности и согласованности.
    """

    # -- response reproducibility (Jaccard) ---------------------------------

    @pytest.mark.slow
    def test_reproducibility_jaccard(self, agent_client, judge_llm):
        """Multiple runs of the same message yield sufficient word overlap. / Несколько запусков одного сообщения дают достаточное совпадение слов."""
        cfg = get_config().stability
        responses: list[set[str]] = []
        # Run same prompt multiple times / Запуск одного промпта несколько раз
        for _ in range(cfg.reproducibility_runs):
            s = AgentSession(client=agent_client, judge=judge_llm)
            s.init_session()
            s.send("Hello, I would like to get started")
            s.expect_response_ok()
            responses.append(set(s.last_text.lower().split()))

        # Compare all pairs with Jaccard similarity / Сравнение всех пар по мере Жаккара
        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                intersection = responses[i] & responses[j]
                union = responses[i] | responses[j]
                jaccard = len(intersection) / len(union) if union else 1.0
                assert jaccard >= cfg.jaccard_min, (
                    f"Runs {i} and {j}: Jaccard={jaccard:.2f} "
                    f"< {cfg.jaccard_min}"
                )

    # -- paraphrase stability -----------------------------------------------

    @pytest.mark.slow
    def test_paraphrase_consistency(self, agent_client):
        """Paraphrased requests yield similarly-sized responses. / Парафразы запросов дают ответы сопоставимого размера."""
        cfg = get_config().stability
        paraphrases = [
            "I need help getting started",
            "Can you help me begin?",
            "I'd like to start the process please",
        ]
        responses: list[str] = []
        for msg in paraphrases:
            s = AgentSession(client=agent_client)
            s.init_session()
            s.send(msg)
            s.expect_response_ok()
            responses.append(s.last_text)

        # Check length ratios stay within bounds / Проверка, что отношение длин в допустимых пределах
        lengths = [len(r) for r in responses]
        avg_len = sum(lengths) / len(lengths)
        for i, length in enumerate(lengths):
            ratio = length / avg_len if avg_len else 1.0
            assert cfg.paraphrase_length_ratio_min <= ratio <= cfg.paraphrase_length_ratio_max, (
                f"Paraphrase {i}: length {length} is too far from "
                f"average {avg_len:.0f} (ratio={ratio:.2f}, "
                f"allowed [{cfg.paraphrase_length_ratio_min}..{cfg.paraphrase_length_ratio_max}])"
            )

    # -- format stability across turns --------------------------------------

    def test_response_format_stable(self, session):
        """Agent maintains a consistent response format across turns. / Агент сохраняет единообразный формат ответов между репликами."""
        session.send("Hello, I need help")
        session.expect_response_ok()
        first_has_question = "?" in session.last_text

        session.send("Can you explain the process?")
        session.expect_response_ok()
        second_has_question = "?" in session.last_text

        session.send("What are the next steps?")
        session.expect_response_ok()
        third_has_question = "?" in session.last_text

        question_count = sum([first_has_question, second_has_question, third_has_question])
        assert question_count >= 1, (
            "Agent never asked a follow-up question across 3 turns"
        )

    # -- no hallucination drift ---------------------------------------------

    @pytest.mark.slow
    def test_no_contradiction_across_runs(self, agent_client, judge_llm):
        """Agent does not contradict itself across independent runs. / Агент не противоречит себе в независимых запусках."""
        cfg = get_config().stability
        responses: list[str] = []
        for _ in range(cfg.reproducibility_runs):
            s = AgentSession(client=agent_client, judge=judge_llm)
            s.init_session()
            s.send("What can you help me with?")
            s.expect_response_ok()
            responses.append(s.last_text.lower())

        # Fail if some runs have "error" and others don't / Ошибка, если в одних запусках есть "error", в других — нет
        error_in_some = any("error" in r for r in responses)
        ok_in_some = any("error" not in r for r in responses)
        if error_in_some and ok_in_some:
            pytest.fail(
                "Inconsistent behavior: some runs contain 'error', others don't.\n"
                + "\n---\n".join(responses)
            )
