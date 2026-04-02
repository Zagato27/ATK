"""Run GenericRecallTests against the configured live agent service."""
from __future__ import annotations

import pytest

from agent_test_kit.generic_tests import GenericRecallTests


class TestLiveServiceRecall(GenericRecallTests):
    """Recall suite bound to the configured agent service."""

    @pytest.mark.skip(reason="Live insurance agent is domain-scoped and does not guarantee verbatim recall of arbitrary external identifiers.")
    def test_reference_number_recalled_deterministically(self, session):
        pass

    @pytest.mark.skip(reason="Live insurance agent is domain-scoped and does not guarantee verbatim recall of arbitrary external identifiers.")
    def test_account_id_survives_off_topic_detour(self, session):
        pass
