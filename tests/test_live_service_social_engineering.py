"""Run GenericSocialEngineeringTests against the configured live agent service."""
from __future__ import annotations

import pytest

from agent_test_kit import AgentSession
from agent_test_kit.generic_tests import GenericSocialEngineeringTests


@pytest.fixture
def session(agent_client, judge_llm):
    """AgentSession wired to the configured agent service and real judge."""
    s = AgentSession(client=agent_client, judge=judge_llm)
    s.init_session()
    return s


class TestLiveServiceSocialEngineering(GenericSocialEngineeringTests):
    """Social-engineering suite bound to the configured agent service."""
