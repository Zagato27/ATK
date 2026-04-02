"""
Unit tests for the Golden Set Regression module.

Uses tmp_path -- no real agent or LLM calls.
"""
import pytest

from agent_test_kit.golden import (
    GoldenCase,
    GoldenReport,
    compare_run,
    load_golden,
    save_golden,
    text_hash,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_CASES = [
    GoldenCase(
        id="greeting",
        input="Привет",
        expected_keywords=["добро пожаловать"],
        baseline_scores={"politeness": 0.85, "relevance": 0.90},
        baseline_text_hash="abc123",
    ),
    GoldenCase(
        id="tariff",
        input="Рассчитай КАСКО",
        expected_keywords=["премия"],
        baseline_scores={"data_extraction": 0.80},
    ),
]


# ---------------------------------------------------------------------------
# Tests: save / load round-trip
# ---------------------------------------------------------------------------

class TestSaveLoad:
    def test_round_trip(self, tmp_path):
        path = tmp_path / "golden.yaml"
        save_golden(path, SAMPLE_CASES)
        loaded = load_golden(path)

        assert len(loaded) == 2
        assert loaded[0].id == "greeting"
        assert loaded[0].input == "Привет"
        assert loaded[0].expected_keywords == ["добро пожаловать"]
        assert loaded[0].baseline_scores == {"politeness": 0.85, "relevance": 0.90}
        assert loaded[0].baseline_text_hash == "abc123"

    def test_load_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_golden(tmp_path / "does_not_exist.yaml")

    def test_load_invalid_format(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text("key: value", encoding="utf-8")
        with pytest.raises(ValueError, match="YAML list"):
            load_golden(path)

    def test_load_item_without_id(self, tmp_path):
        path = tmp_path / "bad_item.yaml"
        path.write_text("- input: hi\n", encoding="utf-8")
        with pytest.raises(ValueError, match="non-empty string 'id'"):
            load_golden(path)

    def test_load_item_not_mapping(self, tmp_path):
        path = tmp_path / "bad_item2.yaml"
        path.write_text("- just-a-string\n", encoding="utf-8")
        with pytest.raises(ValueError, match="must be a mapping"):
            load_golden(path)

    def test_save_creates_dirs(self, tmp_path):
        path = tmp_path / "sub" / "dir" / "golden.yaml"
        save_golden(path, [SAMPLE_CASES[0]])
        assert path.exists()
        loaded = load_golden(path)
        assert len(loaded) == 1

    def test_empty_list(self, tmp_path):
        path = tmp_path / "empty.yaml"
        save_golden(path, [])
        loaded = load_golden(path)
        assert loaded == []


# ---------------------------------------------------------------------------
# Tests: compare_run
# ---------------------------------------------------------------------------

class TestCompareRun:
    def test_all_within_threshold(self):
        current = {
            "greeting": {"politeness": 0.83, "relevance": 0.88},
            "tariff": {"data_extraction": 0.78},
        }
        report = compare_run(current, SAMPLE_CASES, drift_threshold=0.15)
        assert report.all_passed
        assert len(report.items) == 3

    def test_drift_detected(self):
        current = {
            "greeting": {"politeness": 0.40, "relevance": 0.90},
            "tariff": {"data_extraction": 0.80},
        }
        report = compare_run(current, SAMPLE_CASES, drift_threshold=0.15)
        assert not report.all_passed
        drifted = report.drifted
        assert len(drifted) == 1
        assert drifted[0].case_id == "greeting"
        assert drifted[0].metric == "politeness"
        assert drifted[0].drift == pytest.approx(-0.45)

    def test_missing_case_in_results(self):
        current = {
            "greeting": {"politeness": 0.85, "relevance": 0.90},
        }
        report = compare_run(current, SAMPLE_CASES, drift_threshold=0.15)
        assert not report.all_passed
        drifted = report.drifted
        assert any(d.case_id == "tariff" for d in drifted)

    def test_missing_metric_in_results(self):
        current = {
            "greeting": {"politeness": 0.85},
            "tariff": {"data_extraction": 0.80},
        }
        report = compare_run(current, SAMPLE_CASES, drift_threshold=0.15)
        assert not report.all_passed
        drifted = report.drifted
        assert any(d.metric == "relevance" for d in drifted)

    def test_positive_drift_also_flags(self):
        current = {
            "greeting": {"politeness": 1.0, "relevance": 0.90},
            "tariff": {"data_extraction": 1.0},
        }
        report = compare_run(current, SAMPLE_CASES, drift_threshold=0.10)
        drifted = report.drifted
        assert any(d.case_id == "greeting" and d.metric == "politeness" for d in drifted)
        assert any(d.case_id == "tariff" and d.metric == "data_extraction" for d in drifted)

    def test_summary_output(self):
        current = {
            "greeting": {"politeness": 0.40, "relevance": 0.90},
            "tariff": {"data_extraction": 0.80},
        }
        report = compare_run(current, SAMPLE_CASES, drift_threshold=0.15)
        summary = report.summary()
        assert "DRIFT" in summary
        assert "greeting/politeness" in summary


# ---------------------------------------------------------------------------
# Tests: text_hash
# ---------------------------------------------------------------------------

class TestTextHash:
    def test_deterministic(self):
        h1 = text_hash("hello world")
        h2 = text_hash("hello world")
        assert h1 == h2
        assert len(h1) == 16

    def test_different_inputs(self):
        assert text_hash("a") != text_hash("b")


# ---------------------------------------------------------------------------
# Tests: GoldenReport
# ---------------------------------------------------------------------------

class TestGoldenReport:
    def test_empty_report(self):
        r = GoldenReport()
        assert r.all_passed
        assert r.drifted == []
