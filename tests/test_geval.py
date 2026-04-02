"""
Unit tests for ATK G-Eval module.

Uses a mock judge -- no real LLM calls.
"""
import logging

import pytest

from agent_test_kit.geval import ATKGEval, GEvalResult, _COT_CACHE
from agent_test_kit.judge import BaseLLMJudge


# ---------------------------------------------------------------------------
# Mock judge
# ---------------------------------------------------------------------------

class MockJudge(BaseLLMJudge):
    """Returns canned responses. Tracks prompts received."""

    def __init__(self, responses: list[str]):
        self._responses = responses
        self._idx = 0
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        resp = self._responses[self._idx % len(self._responses)]
        self._idx += 1
        return resp

    def get_model_name(self) -> str:
        return "mock-judge"


@pytest.fixture(autouse=True)
def _clear_cot_cache():
    _COT_CACHE.clear()
    yield
    _COT_CACHE.clear()


# ---------------------------------------------------------------------------
# Tests: CoT generation
# ---------------------------------------------------------------------------

class TestCoTGeneration:
    def test_generates_steps(self):
        judge = MockJudge([
            "1. Прочитайте сообщение.\n2. Оцените вежливость.\n3. Выставьте оценку."
        ])
        geval = ATKGEval(judge=judge)
        steps = geval.generate_evaluation_steps("Оцените вежливость ответа")
        assert len(steps) == 3
        assert "Прочитайте" in steps[0]

    def test_caches_in_memory(self):
        judge = MockJudge([
            "1. Step A.\n2. Step B.",
            "1. Different.\n2. Steps.",
        ])
        geval = ATKGEval(judge=judge)
        criteria = "test criteria"
        steps1 = geval.generate_evaluation_steps(criteria)
        steps2 = geval.generate_evaluation_steps(criteria)
        assert steps1 == steps2
        assert len(judge.prompts) == 1

    def test_fallback_on_empty(self):
        judge = MockJudge(["Here is some text without numbered steps."])
        geval = ATKGEval(judge=judge)
        steps = geval.generate_evaluation_steps("criteria")
        assert len(steps) >= 3
        assert any("1." in s for s in steps)

    def test_caches_on_disk(self, tmp_path):
        judge = MockJudge(["1. Step one.\n2. Step two."])
        geval = ATKGEval(judge=judge, cot_cache_dir=str(tmp_path / "cot"))
        steps = geval.generate_evaluation_steps("disk test")
        assert len(steps) == 2

        _COT_CACHE.clear()
        geval2 = ATKGEval(judge=judge, cot_cache_dir=str(tmp_path / "cot"))
        steps2 = geval2.generate_evaluation_steps("disk test")
        assert steps2 == steps
        assert len(judge.prompts) == 1


# ---------------------------------------------------------------------------
# Tests: score parsing
# ---------------------------------------------------------------------------

class TestScoreParsing:
    def test_parses_dash_format(self):
        geval = ATKGEval(judge=MockJudge([]))
        assert geval._parse_score("- quality: 4") == 4

    def test_parses_colon_format(self):
        geval = ATKGEval(judge=MockJudge([]))
        assert geval._parse_score("quality: 3") == 3

    def test_parses_bare_number(self):
        geval = ATKGEval(judge=MockJudge([]))
        assert geval._parse_score("The score is 5") == 5

    def test_ignores_out_of_range(self):
        geval = ATKGEval(judge=MockJudge([]), score_scale=5)
        assert geval._parse_score("Score: 99") is None

    def test_returns_none_on_garbage(self):
        geval = ATKGEval(judge=MockJudge([]))
        assert geval._parse_score("no numbers here") is None

    def test_prefers_metric_line_over_other_numbers(self):
        geval = ATKGEval(judge=MockJudge([]))
        response = (
            "В 2026 году было 14 обращений.\n"
            "- quality: 2\n"
            "Комментарий: 5/5"
        )
        assert geval._parse_score(response, metric_name="quality") == 2

    def test_extract_reasoning_from_reasoning_line(self):
        geval = ATKGEval(judge=MockJudge([]))
        response = "- quality: 4\n- reasoning: Короткое и четкое обоснование."
        reasoning = geval._extract_reasoning(response, metric_name="quality")
        assert "обоснование" in reasoning.lower()


# ---------------------------------------------------------------------------
# Tests: full evaluate pipeline
# ---------------------------------------------------------------------------

class TestEvaluate:
    def test_single_sample_pass(self):
        judge = MockJudge([
            "1. Read.\n2. Evaluate.\n3. Score.",
            "- quality: 4",
        ])
        geval = ATKGEval(judge=judge, n_samples=1, score_scale=5)
        result = geval.evaluate(
            input_text="Привет",
            output_text="Здравствуйте! Чем могу помочь?",
            criteria="Оцените вежливость ответа",
            threshold=0.5,
        )
        assert isinstance(result, GEvalResult)
        assert result.passed
        assert result.score == pytest.approx(0.75)
        assert result.raw_scores == [4]
        assert len(result.evaluation_steps) == 3

    def test_single_sample_fail(self):
        judge = MockJudge([
            "1. Read.\n2. Evaluate.\n3. Score.",
            "- quality: 2",
        ])
        geval = ATKGEval(judge=judge, n_samples=1, score_scale=5)
        result = geval.evaluate(
            input_text="hi", output_text="go away",
            criteria="politeness", threshold=0.8,
        )
        assert not result.passed
        assert result.score == pytest.approx(0.25)

    def test_multi_sample_averaging(self):
        judge = MockJudge([
            "1. Step.\n2. Step.\n3. Step.",
            "- quality: 3",
            "- quality: 5",
            "- quality: 4",
        ])
        geval = ATKGEval(judge=judge, n_samples=3, score_scale=5)
        result = geval.evaluate(
            input_text="q", output_text="a",
            criteria="test", threshold=0.3,
        )
        assert result.passed
        assert len(result.raw_scores) == 3
        assert result.raw_scores == [3, 5, 4]
        expected_avg = (3 + 5 + 4) / 3
        expected_norm = (expected_avg - 1) / 4
        assert result.score == pytest.approx(expected_norm)

    def test_all_samples_fail(self):
        judge = MockJudge([
            "1. Step.\n2. Step.\n3. Step.",
            "unparseable garbage",
        ])
        geval = ATKGEval(judge=judge, n_samples=1, score_scale=5)
        result = geval.evaluate(
            input_text="q", output_text="a",
            criteria="test", threshold=0.5,
        )
        assert not result.passed
        assert result.score == 0.0
        assert result.raw_scores == []

    def test_context_included_in_prompt(self):
        judge = MockJudge([
            "1. Step.\n2. Step.\n3. Step.",
            "- quality: 5",
        ])
        geval = ATKGEval(judge=judge, n_samples=1)
        result = geval.evaluate(
            input_text="q", output_text="a",
            criteria="test", threshold=0.5,
            context=["Факт 1", "Факт 2"],
        )
        eval_prompt = judge.prompts[1]
        assert "Контекст:" in eval_prompt
        assert "Факт 1" in eval_prompt

    def test_expected_output_in_prompt(self):
        judge = MockJudge([
            "1. Step.\n2. Step.\n3. Step.",
            "- quality: 4",
        ])
        geval = ATKGEval(judge=judge, n_samples=1)
        result = geval.evaluate(
            input_text="q", output_text="a",
            criteria="test", threshold=0.5,
            expected_output="expected answer",
        )
        eval_prompt = judge.prompts[1]
        assert "Ожидаемый ответ:" in eval_prompt
        assert "expected answer" in eval_prompt

    def test_reasoning_backfill_when_score_only(self):
        judge = MockJudge([
            "1. Step.\n2. Step.\n3. Step.",
            "- quality: 4",
            "- reasoning: Ответ вежливый, профессиональный и релевантный.",
        ])
        geval = ATKGEval(judge=judge, n_samples=1, require_reasoning=True)
        result = geval.evaluate(
            input_text="q", output_text="a",
            criteria="test", threshold=0.5,
            metric_name="quality",
        )
        assert result.passed
        assert "вежливый" in result.reasoning.lower()
        assert len(result.raw_responses) == 2

    def test_fails_when_reasoning_required_but_missing(self):
        judge = MockJudge([
            "1. Step.\n2. Step.\n3. Step.",
            "- quality: 4",
            "- quality: 4",
            "- quality: 4",
        ])
        geval = ATKGEval(
            judge=judge,
            n_samples=1,
            require_reasoning=True,
            reasoning_backfill_attempts=2,
        )
        result = geval.evaluate(
            input_text="q", output_text="a",
            criteria="test", threshold=0.5,
            metric_name="quality",
        )
        assert not result.passed
        assert result.reasoning == "Обоснование не предоставлено моделью."


# ---------------------------------------------------------------------------
# Tests: prompt construction
# ---------------------------------------------------------------------------

class TestPromptConstruction:
    def test_russian_template(self):
        judge = MockJudge([
            "1. Шаг.\n2. Шаг.\n3. Шаг.",
            "- метрика: 3",
        ])
        geval = ATKGEval(judge=judge, n_samples=1)
        geval.evaluate(
            input_text="тест", output_text="ответ",
            criteria="критерий", threshold=0.5,
            metric_name="метрика",
        )
        prompt = judge.prompts[1]
        assert "[SYSTEM]" in prompt
        assert "экспертный оценщик качества ответов AI-агентов" in prompt
        assert "Вам будет предоставлен один ответ AI-агента" in prompt
        assert "Критерии оценки:" in prompt
        assert "Шаги оценивания:" in prompt
        assert "Сообщение пользователя:" in prompt
        assert "Ответ агента:" in prompt
        assert "Форма оценки (ТОЛЬКО оценки):" in prompt

    def test_system_prompt_version(self):
        geval = ATKGEval(
            judge=MockJudge([]),
            system_prompt_version="v1",
        )
        assert "экспертный оценщик" in geval.system_prompt


class TestVerboseLogging:
    def test_verbose_logs_include_criteria_chat_and_reasoning(self, caplog):
        judge = MockJudge([
            "1. Проверить критерии.\n2. Оценить ответ.\n3. Выставить балл.",
            "- quality: 4\nПричина: ответ вежливый и по теме.",
        ])
        geval = ATKGEval(
            judge=judge,
            n_samples=1,
            verbose_logging=True,
        )
        with caplog.at_level(logging.DEBUG, logger="agent_test_kit.geval"):
            result = geval.evaluate(
                input_text="Привет",
                output_text="Здравствуйте! Чем могу помочь?",
                criteria="Оценить вежливость, релевантность и ясность ответа.",
                threshold=0.5,
                metric_name="quality",
            )
        logs = "\n".join(record.message for record in caplog.records)
        assert "ATKGEval criteria" in logs
        assert "ATKGEval chat user" in logs
        assert "ATKGEval chat agent" in logs
        assert "ATKGEval raw judge response" in logs
        assert "ATKGEval reasoning" in logs
        assert result.passed


# ---------------------------------------------------------------------------
# Tests: GEvalResult dataclass
# ---------------------------------------------------------------------------

class TestGEvalResult:
    def test_defaults(self):
        r = GEvalResult(score=0.5)
        assert r.raw_scores == []
        assert not r.passed
        assert r.reasoning == ""
        assert r.evaluation_steps == []
        assert r.raw_responses == []
