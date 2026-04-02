"""Run GenericReproducibilityTests against the configured live agent service."""
from __future__ import annotations

from agent_test_kit.generic_tests import GenericReproducibilityTests


class TestLiveServiceReproducibility(GenericReproducibilityTests):
    """Reproducibility suite bound to the configured agent service."""
