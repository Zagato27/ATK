"""Run GenericScopeRecoveryTests against the configured live agent service."""
from __future__ import annotations

import pytest

from agent_test_kit import AgentSession
from agent_test_kit.generic_tests import GenericScopeRecoveryTests


@pytest.fixture
def session(agent_client, judge_llm):
    """AgentSession wired to the configured agent service and real judge."""
    s = AgentSession(client=agent_client, judge=judge_llm)
    s.init_session()
    return s


class TestLiveServiceScopeRecovery(GenericScopeRecoveryTests):
    """Scope-recovery suite bound to the configured agent service."""
