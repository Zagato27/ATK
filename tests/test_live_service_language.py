"""Run GenericLanguageTests against the configured live agent service."""
from __future__ import annotations

from agent_test_kit.generic_tests import GenericLanguageTests


class TestLiveServiceLanguage(GenericLanguageTests):
    """Language-policy suite bound to the configured agent service."""
