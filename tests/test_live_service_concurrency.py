"""Run GenericConcurrencyTests against the configured live agent service."""
from __future__ import annotations

from agent_test_kit.generic_tests import GenericConcurrencyTests


class TestLiveServiceConcurrency(GenericConcurrencyTests):
    """Concurrency suite bound to the configured agent service."""
