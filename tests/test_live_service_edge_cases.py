"""Run GenericEdgeCaseTests against a live local HTTP service."""
from __future__ import annotations

from agent_test_kit.generic_tests import GenericEdgeCaseTests


class TestLiveServiceEdgeCases(GenericEdgeCaseTests):
    """Generic edge-case suite bound to the configured agent service."""
