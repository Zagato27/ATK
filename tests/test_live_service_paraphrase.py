"""Run GenericParaphraseConsistencyTests against the configured live agent service."""
from __future__ import annotations

from agent_test_kit.generic_tests import GenericParaphraseConsistencyTests


class TestLiveServiceParaphrase(GenericParaphraseConsistencyTests):
    """Paraphrase-consistency suite bound to the configured agent service."""
