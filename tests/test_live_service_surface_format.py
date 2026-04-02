"""Run GenericSurfaceFormatTests against the configured live agent service."""
from __future__ import annotations

from agent_test_kit.generic_tests import GenericSurfaceFormatTests


class TestLiveServiceSurfaceFormat(GenericSurfaceFormatTests):
    """Surface-format suite bound to the configured agent service."""
