"""Run GenericCorrectionTests against the configured live agent service."""
from __future__ import annotations

import pytest

from agent_test_kit import AgentSession
from agent_test_kit.generic_tests import GenericCorrectionTests


@pytest.fixture
def session(agent_client, judge_llm):
    """AgentSession wired to the configured agent service and real judge."""
    s = AgentSession(client=agent_client, judge=judge_llm)
    s.init_session()
    return s


class TestLiveServiceCorrections(GenericCorrectionTests):
    """Correction suite bound to the configured agent service."""

    @pytest.mark.skip(reason="Live insurance agent keeps the business flow and does not support generic city-memory correction outside its domain scenario.")
    def test_corrected_city_overrides_previous_value(self, session):
        pass
