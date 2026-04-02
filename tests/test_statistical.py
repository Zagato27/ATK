"""
Unit tests for the statistical testing module.

Uses pure in-memory data -- no network, no LLM.
"""
import pytest

from agent_test_kit.statistical import (
    Distribution,
    RunResult,
    mann_whitney_u,
    run_n_times,
)


# ---------------------------------------------------------------------------
# Distribution basics
# ---------------------------------------------------------------------------

class TestDistribution:
    def test_empty(self):
        d = Distribution()
        assert d.n == 0
        assert d.pass_rate == 0.0
        assert d.mean_score is None
        assert d.std_score is None
        assert d.mean_latency == 0.0

    def test_all_passed(self):
        d = Distribution(results=[
            RunResult(passed=True, score=0.9, latency=1.0),
            RunResult(passed=True, score=0.8, latency=2.0),
        ])
        assert d.n == 2
        assert d.pass_rate == 1.0
        assert d.mean_score == pytest.approx(0.85)
        assert d.mean_latency == pytest.approx(1.5)

    def test_mixed_pass_fail(self):
        d = Distribution(results=[
            RunResult(passed=True, score=1.0, latency=0.5),
            RunResult(passed=False, score=0.0, latency=0.5),
            RunResult(passed=True, score=0.8, latency=0.5),
        ])
        assert d.pass_rate == pytest.approx(2 / 3)

    def test_std_score(self):
        d = Distribution(results=[
            RunResult(passed=True, score=0.5, latency=0.1),
            RunResult(passed=True, score=0.5, latency=0.1),
        ])
        assert d.std_score == pytest.approx(0.0)

    def test_std_score_single(self):
        d = Distribution(results=[
            RunResult(passed=True, score=0.5, latency=0.1),
        ])
        assert d.std_score is None

    def test_scores_ignores_none(self):
        d = Distribution(results=[
            RunResult(passed=True, score=None, latency=0.1),
            RunResult(passed=True, score=0.7, latency=0.1),
        ])
        assert d.scores == [0.7]


class TestConfidenceInterval:
    def test_single_value(self):
        d = Distribution(results=[
            RunResult(passed=True, score=0.9, latency=0.1),
        ])
        lo, hi = d.confidence_interval()
        assert lo == pytest.approx(0.9)
        assert hi == pytest.approx(0.9)

    def test_narrow_distribution(self):
        d = Distribution(results=[
            RunResult(passed=True, score=0.9, latency=0.1),
            RunResult(passed=True, score=0.9, latency=0.1),
            RunResult(passed=True, score=0.9, latency=0.1),
        ])
        lo, hi = d.confidence_interval()
        assert lo == pytest.approx(0.9)
        assert hi == pytest.approx(0.9)

    def test_wide_distribution(self):
        d = Distribution(results=[
            RunResult(passed=True, score=0.2, latency=0.1),
            RunResult(passed=True, score=0.4, latency=0.1),
            RunResult(passed=True, score=0.6, latency=0.1),
            RunResult(passed=True, score=0.8, latency=0.1),
            RunResult(passed=True, score=1.0, latency=0.1),
        ])
        lo, hi = d.confidence_interval()
        assert lo < hi
        assert lo >= 0.0
        assert hi <= 1.0

    def test_empty(self):
        d = Distribution()
        lo, hi = d.confidence_interval()
        assert lo == 0.0
        assert hi == 0.0


class TestIsStable:
    def test_stable(self):
        d = Distribution(results=[
            RunResult(passed=True, score=0.9, latency=0.1),
            RunResult(passed=True, score=0.85, latency=0.1),
        ])
        assert d.is_stable(min_pass_rate=0.8, max_std=0.2)

    def test_unstable_pass_rate(self):
        d = Distribution(results=[
            RunResult(passed=True, score=0.9, latency=0.1),
            RunResult(passed=False, score=0.1, latency=0.1),
        ])
        assert not d.is_stable(min_pass_rate=0.8, max_std=0.5)


# ---------------------------------------------------------------------------
# run_n_times
# ---------------------------------------------------------------------------

class TestRunNTimes:
    def test_sequential(self):
        counter = {"n": 0}

        def fn():
            counter["n"] += 1
            return RunResult(passed=True, score=0.9, latency=0.01)

        dist = run_n_times(fn, 5)
        assert dist.n == 5
        assert counter["n"] == 5
        assert dist.pass_rate == 1.0

    def test_parallel(self):
        def fn():
            return RunResult(passed=True, score=0.8, latency=0.01)

        dist = run_n_times(fn, 4, parallel=True)
        assert dist.n == 4

    def test_exception_captured(self):
        call_count = {"n": 0}

        def fn():
            call_count["n"] += 1
            if call_count["n"] == 2:
                raise RuntimeError("boom")
            return RunResult(passed=True, latency=0.01)

        dist = run_n_times(fn, 3)
        assert dist.n == 3
        failed = [r for r in dist.results if not r.passed]
        assert len(failed) == 1
        assert "boom" in failed[0].error


# ---------------------------------------------------------------------------
# mann_whitney_u
# ---------------------------------------------------------------------------

class TestMannWhitneyU:
    def test_identical_distributions(self):
        d = Distribution(results=[
            RunResult(passed=True, score=0.5, latency=0.1),
            RunResult(passed=True, score=0.5, latency=0.1),
            RunResult(passed=True, score=0.5, latency=0.1),
        ])
        p = mann_whitney_u(d, d)
        assert p > 0.05

    def test_very_different_distributions(self):
        da = Distribution(results=[
            RunResult(passed=True, score=0.1, latency=0.1),
            RunResult(passed=True, score=0.15, latency=0.1),
            RunResult(passed=True, score=0.12, latency=0.1),
            RunResult(passed=True, score=0.08, latency=0.1),
        ])
        db = Distribution(results=[
            RunResult(passed=True, score=0.9, latency=0.1),
            RunResult(passed=True, score=0.95, latency=0.1),
            RunResult(passed=True, score=0.88, latency=0.1),
            RunResult(passed=True, score=0.92, latency=0.1),
        ])
        p = mann_whitney_u(da, db)
        assert p < 0.05

    def test_too_few_scores(self):
        da = Distribution(results=[RunResult(passed=True, score=0.5, latency=0.1)])
        db = Distribution(results=[RunResult(passed=True, score=0.5, latency=0.1)])
        with pytest.raises(ValueError, match="Need >= 2"):
            mann_whitney_u(da, db)


# ---------------------------------------------------------------------------
# AgentSession integration (expect_pass_rate / expect_score_ci)
# ---------------------------------------------------------------------------

class TestSessionStatistical:
    def test_expect_pass_rate_passes(self):
        from agent_test_kit import AgentSession
        d = Distribution(results=[
            RunResult(passed=True, latency=0.1) for _ in range(9)
        ] + [RunResult(passed=False, latency=0.1)])
        AgentSession.expect_pass_rate(d, 0.8)

    def test_expect_pass_rate_fails(self):
        from agent_test_kit import AgentSession
        d = Distribution(results=[
            RunResult(passed=True, latency=0.1),
            RunResult(passed=False, latency=0.1),
        ])
        with pytest.raises(AssertionError, match="Pass rate"):
            AgentSession.expect_pass_rate(d, 0.9)

    def test_expect_score_ci_passes(self):
        from agent_test_kit import AgentSession
        d = Distribution(results=[
            RunResult(passed=True, score=0.9, latency=0.1),
            RunResult(passed=True, score=0.85, latency=0.1),
            RunResult(passed=True, score=0.88, latency=0.1),
            RunResult(passed=True, score=0.92, latency=0.1),
            RunResult(passed=True, score=0.87, latency=0.1),
        ])
        AgentSession.expect_score_ci(d, min_lower_bound=0.7)

    def test_expect_score_ci_fails(self):
        from agent_test_kit import AgentSession
        d = Distribution(results=[
            RunResult(passed=True, score=0.3, latency=0.1),
            RunResult(passed=True, score=0.35, latency=0.1),
            RunResult(passed=True, score=0.28, latency=0.1),
        ])
        with pytest.raises(AssertionError, match="CI lower bound"):
            AgentSession.expect_score_ci(d, min_lower_bound=0.9)
