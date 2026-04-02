"""Run GenericPayloadSafetyTests against the configured live agent service."""
from __future__ import annotations

from agent_test_kit.generic_tests import GenericPayloadSafetyTests


class TestLiveServicePayloadSafety(GenericPayloadSafetyTests):
    """Payload-safety suite bound to the configured agent service."""
