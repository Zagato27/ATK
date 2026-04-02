"""
Integration tests that go through the real LLM judge.

Requires a configured API key in agent-test-kit.toml (judge.api_key)
or via environment variable (judge.api_key_env).

Run:  python -m pytest tests/test_integration.py -v --log-cli-level=DEBUG
Skip: python -m pytest tests -q --ignore=tests/test_integration.py
"""
import pytest

pytestmark = pytest.mark.judge


def _is_real_judge(judge_llm) -> bool:
    return judge_llm is not None


@pytest.fixture
def session(agent_client, judge_llm):
    from agent_test_kit import AgentSession

    s = AgentSession(client=agent_client, judge=judge_llm)
    s.init_session()
    return s


class TestRealJudgeEvaluation:
    """Evaluate() through real LLM judge (MiniMax / Anthropic / OpenAI)."""

    def test_politeness_positive(self, session):
        """Polite greeting should score well."""
        if not _is_real_judge(session._judge):
            pytest.skip("Real judge not configured — set judge.api_key")

        session.send("Привет!")
        session.evaluate("politeness", threshold=0.5)

        result = session.last_eval_result
        assert result is not None
        assert result.score >= 0.5
        assert result.reasoning, "Reasoning must not be empty"

    def test_context_retention(self, session):
        """Agent should retain context across turns."""
        if not _is_real_judge(session._judge):
            pytest.skip("Real judge not configured — set judge.api_key")

        session.send("Меня зовут Павел")
        session.send("Как меня зовут?")
        session.evaluate("context_retention", threshold=0.4)

        result = session.last_eval_result
        assert result is not None
        assert result.score >= 0.4
        assert result.reasoning, "Reasoning must not be empty"

    def test_evaluate_direct_returns_detailed_result(self, session):
        """evaluate_direct() should return GEvalResult with all fields."""
        if not _is_real_judge(session._judge):
            pytest.skip("Real judge not configured — set judge.api_key")

        session.send("Добрый день!")
        session.evaluate_direct("politeness", threshold=0.3)

        r = session.last_eval_result
        assert r is not None
        assert 0.0 <= r.score <= 1.0
        assert len(r.raw_scores) >= 1
        assert r.evaluation_steps
        assert r.reasoning
