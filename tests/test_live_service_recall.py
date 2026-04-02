"""Run GenericRecallTests against the configured live agent service."""
from __future__ import annotations

from agent_test_kit.generic_tests import GenericRecallTests


class TestLiveServiceRecall(GenericRecallTests):
    """Recall suite bound to the configured agent service."""
