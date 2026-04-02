"""Run GenericSessionResilienceTests against the configured live agent service."""
from __future__ import annotations

from agent_test_kit.generic_tests import GenericSessionResilienceTests


class TestLiveServiceSessionResilience(GenericSessionResilienceTests):
    """Session-resilience suite bound to the configured agent service."""
