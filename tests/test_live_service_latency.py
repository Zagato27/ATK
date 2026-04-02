"""Run GenericLatencyTests against the configured live agent service."""
from __future__ import annotations

from agent_test_kit.generic_tests import GenericLatencyTests


class TestLiveServiceLatency(GenericLatencyTests):
    """Latency suite bound to the configured agent service."""
