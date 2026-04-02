"""
Unit tests for AgentSession deterministic assertions.

Uses MockClient — no network, no LLM judge.
"""
import pytest

from agent_test_kit import AgentResponse, AgentSession, BaseAgentClient, BaseLLMJudge
from agent_test_kit.geval import GEvalResult
from agent_test_kit.metrics import MetricRegistry
from agent_test_kit.session import (
    _install_session_observer,
    _restore_session_observer,
    run_dialogue,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _Client(BaseAgentClient):
    """Tiny mock that returns one fixed response per send."""

    def __init__(
        self,
        replies: list[str],
        tool_calls: list[list[dict]] | None = None,
    ):
        super().__init__(base_url="http://test", timeout=1, verify=False)
        self._replies = replies
        self._tool_calls = tool_calls
        self._idx = 0

    def create_session(self, **kw):
        self.session_id = "s1"
        return {"session_id": "s1", "message": "Welcome!"}

    def send_message(self, message, **kw):
        text = self._replies[self._idx % len(self._replies)]
        tc = (
            self._tool_calls[self._idx % len(self._tool_calls)]
            if self._tool_calls
            else [{"name": "calc_tariff"}]
        )
        self._idx += 1
        return AgentResponse(
            text=text, status_code=200,
            metadata={"node": "main", "tool_calls": tc},
            raw={"response": text, "node": "main"},
        )

    def reset(self):
        super().reset()
        self._idx = 0


class _RuleJudge(BaseLLMJudge):
    """Prompt-aware mock judge for deterministic semantic tests."""

    def generate(self, prompt: str) -> str:
        if "Сгенерируйте пошаговую инструкцию" in prompt:
            return "1. Прочитайте вопрос.\n2. Прочитайте ответ.\n3. Выставьте оценку."
        if "Формат ответа:" in prompt and "- reasoning:" in prompt:
            return "- reasoning: Ответ вежливый и корректный по формату."
        return "- politeness: 4"

    def get_model_name(self) -> str:
        return "rule-judge"


def _session(
    *replies: str,
    tool_calls: list[list[dict]] | None = None,
    judge: BaseLLMJudge | None = None,
) -> AgentSession:
    client = _Client(list(replies), tool_calls=tool_calls)
    s = AgentSession(client=client, judge=judge)
    s.init_session()
    return s


# ---------------------------------------------------------------------------
# Tests: lifecycle
# ---------------------------------------------------------------------------

class TestLifecycle:
    def test_init_session_stores_data(self):
        s = _session("ok")
        assert s.init_data["session_id"] == "s1"
        assert s.init_message == "Welcome!"
        assert s.turn == 0

    def test_send_increments_turn(self):
        s = _session("reply")
        s.send("hi")
        assert s.turn == 1
        assert len(s.history) == 3  # init assistant + user + assistant

    def test_reset_clears_state(self):
        s = _session("reply")
        s.send("hi")
        s.reset()
        assert s.turn == 0
        assert s.init_message == "Welcome!"

    def test_session_observer_notified_on_creation(self):
        seen: list[AgentSession] = []
        previous = _install_session_observer(seen.append)
        try:
            created = AgentSession(client=_Client(["ok"]))
        finally:
            _restore_session_observer(previous)
        assert seen == [created]


class TestDiagnostics:
    def test_to_trace_dict_contains_response_and_eval(self):
        s = _session("Helpful answer")
        s.send("hello")
        s._last_eval_result = GEvalResult(  # noqa: SLF001 - targeted serialization test
            score=0.9,
            raw_scores=[4, 5],
            passed=True,
            reasoning="Looks good.",
            evaluation_steps=["Read the answer", "Score the answer"],
            raw_responses=["- politeness: 5"],
        )

        trace = s.to_trace_dict()

        assert trace["session_id"] == "s1"
        assert trace["turn"] == 1
        assert trace["last_response"]["text"] == "Helpful answer"
        assert trace["last_eval_result"]["score"] == 0.9
        assert trace["last_eval_result"]["reasoning"] == "Looks good."


# ---------------------------------------------------------------------------
# Tests: expect_response_ok
# ---------------------------------------------------------------------------

class TestExpectResponseOk:
    def test_passes_on_200_with_text(self):
        s = _session("Good response")
        s.send("hi")
        s.expect_response_ok()

    def test_fails_on_empty_text(self):
        s = _session("   ")
        s.send("hi")
        with pytest.raises(AssertionError, match="empty"):
            s.expect_response_ok()

    def test_fails_without_send(self):
        s = _session("x")
        with pytest.raises(AssertionError, match="Call send"):
            s.expect_response_ok()


# ---------------------------------------------------------------------------
# Tests: keyword assertions
# ---------------------------------------------------------------------------

class TestKeywords:
    def test_expect_contains_passes(self):
        s = _session("Hello World")
        s.send("x").expect_contains("hello", "world")

    def test_expect_contains_fails(self):
        s = _session("Hello World")
        s.send("x")
        with pytest.raises(AssertionError, match="missing"):
            s.expect_contains("goodbye")

    def test_expect_contains_any_passes(self):
        s = _session("Hello World")
        s.send("x").expect_contains_any("goodbye", "hello")

    def test_expect_contains_any_fails(self):
        s = _session("Hello World")
        s.send("x")
        with pytest.raises(AssertionError, match="none of"):
            s.expect_contains_any("goodbye", "nope")

    def test_expect_not_contains_passes(self):
        s = _session("Hello World")
        s.send("x").expect_not_contains("error", "sql")

    def test_expect_not_contains_fails(self):
        s = _session("Hello World")
        s.send("x")
        with pytest.raises(AssertionError, match="forbidden"):
            s.expect_not_contains("hello")


# ---------------------------------------------------------------------------
# Tests: latency & length
# ---------------------------------------------------------------------------

class TestLatencyAndLength:
    def test_expect_latency_under_passes(self):
        s = _session("fast")
        s.send("x")
        s.expect_latency_under(10.0)

    def test_expect_latency_under_no_timings(self):
        s = _session("x")
        with pytest.raises(AssertionError, match="No timings"):
            s.expect_latency_under(1.0)

    def test_expect_response_length_passes(self):
        s = _session("12345")
        s.send("x")
        s.expect_response_length(min_chars=1, max_chars=100)

    def test_expect_response_length_fails_too_short(self):
        s = _session("hi")
        s.send("x")
        with pytest.raises(AssertionError, match="length"):
            s.expect_response_length(min_chars=100)


# ---------------------------------------------------------------------------
# Tests: format checks
# ---------------------------------------------------------------------------

class TestFormat:
    def test_expect_asks_question_passes(self):
        s = _session("How are you?")
        s.send("x").expect_asks_question()

    def test_expect_asks_question_fails(self):
        s = _session("No question here.")
        s.send("x")
        with pytest.raises(AssertionError, match="\\?"):
            s.expect_asks_question()

    def test_expect_formal_you_passes(self):
        s = _session("Как Вас зовут?")
        s.send("x").expect_formal_you()

    def test_expect_formal_you_fails(self):
        s = _session("Как ты себя чувствуешь?")
        s.send("x")
        with pytest.raises(AssertionError, match="informal"):
            s.expect_formal_you()


# ---------------------------------------------------------------------------
# Tests: PII
# ---------------------------------------------------------------------------

class TestPII:
    def test_expect_no_pii_passes(self):
        s = _session("No sensitive data here")
        s.send("x").expect_no_pii()

    def test_expect_no_pii_fails(self):
        s = _session("Passport 4515 123456")
        s.send("x")
        with pytest.raises(AssertionError, match="PII"):
            s.expect_no_pii()

    def test_custom_pii_pattern(self):
        s = _session("email: user@example.com")
        s.send("x")
        with pytest.raises(AssertionError, match="PII"):
            s.expect_no_pii(patterns=[r"\S+@\S+\.\S+"])


# ---------------------------------------------------------------------------
# Tests: metadata & raw
# ---------------------------------------------------------------------------

class TestMetadata:
    def test_expect_metadata_passes(self):
        s = _session("ok")
        s.send("x").expect_metadata("node", "main")

    def test_expect_metadata_key_missing(self):
        s = _session("ok")
        s.send("x")
        with pytest.raises(AssertionError, match="missing"):
            s.expect_metadata("nonexistent")

    def test_expect_raw_field_passes(self):
        s = _session("ok")
        s.send("x").expect_raw_field("response", "ok")


# ---------------------------------------------------------------------------
# Tests: init checks
# ---------------------------------------------------------------------------

class TestInitChecks:
    def test_expect_init_contains_passes(self):
        s = _session("ok")
        s.expect_init_contains("welcome")

    def test_expect_init_contains_fails(self):
        s = _session("ok")
        with pytest.raises(AssertionError, match="missing"):
            s.expect_init_contains("nonexistent")

    def test_expect_init_data_passes(self):
        s = _session("ok")
        s.expect_init_data("session_id", "s1")


# ---------------------------------------------------------------------------
# Tests: run_dialogue helper
# ---------------------------------------------------------------------------

class TestRunDialogue:
    def test_run_dialogue_sends_all(self):
        s = _session("reply")
        run_dialogue(s, ["a", "b", "c"])
        assert s.turn == 3

    def test_run_dialogue_without_ok(self):
        s = _session("   ")  # empty — would fail expect_response_ok
        run_dialogue(s, ["a"], expect_ok=False)
        assert s.turn == 1


# ---------------------------------------------------------------------------
# Tests: MetricRegistry
# ---------------------------------------------------------------------------

class TestMetricRegistry:
    def test_register_and_get(self):
        reg = MetricRegistry()
        reg.register("test", "criteria text")
        assert reg.get("test") == "criteria text"

    def test_get_missing_raises(self):
        reg = MetricRegistry()
        with pytest.raises(KeyError, match="not found"):
            reg.get("missing")

    def test_register_bulk(self):
        reg = MetricRegistry()
        reg.register_bulk({"a": "A", "b": "B"})
        assert reg.names() == ["a", "b"]
        assert len(reg) == 2

    def test_contains(self):
        reg = MetricRegistry()
        reg.register("x", "X")
        assert reg.contains("x")
        assert not reg.contains("y")


# ---------------------------------------------------------------------------
# Tests: AgentResponse
# ---------------------------------------------------------------------------

class TestAgentResponse:
    def test_from_raw_picks_first_text_key(self):
        from agent_test_kit import AgentResponse
        resp = AgentResponse.from_raw({"response": "hello", "node": "main"})
        assert resp.text == "hello"
        assert resp.metadata == {"node": "main"}

    def test_from_raw_fallback(self):
        from agent_test_kit import AgentResponse
        resp = AgentResponse.from_raw({"data": 123})
        assert "123" in resp.text  # falls back to str(data)

    def test_from_raw_tries_message_key(self):
        from agent_test_kit import AgentResponse
        resp = AgentResponse.from_raw({"message": "hi"})
        assert resp.text == "hi"


# ---------------------------------------------------------------------------
# Tests: fluent chaining
# ---------------------------------------------------------------------------

class TestChaining:
    def test_fluent_chain(self):
        s = _session("Hello World! How are you?")
        result = (
            s.send("hi")
             .expect_response_ok()
             .expect_contains("hello")
             .expect_not_contains("error")
             .expect_asks_question()
             .expect_response_length(min_chars=1, max_chars=500)
        )
        assert result is s


# ---------------------------------------------------------------------------
# Tests: tool call assertions
# ---------------------------------------------------------------------------

class TestToolCalls:
    def test_expect_tool_called_passes(self):
        s = _session("ok")
        s.send("x").expect_tool_called("calc_tariff")

    def test_expect_tool_called_case_insensitive(self):
        s = _session("ok")
        s.send("x").expect_tool_called("CALC_TARIFF")

    def test_expect_tool_called_fails(self):
        s = _session("ok")
        s.send("x")
        with pytest.raises(AssertionError, match="expected tool"):
            s.expect_tool_called("nonexistent_tool")

    def test_expect_tool_not_called_passes(self):
        s = _session("ok")
        s.send("x").expect_tool_not_called("delete_user", "drop_table")

    def test_expect_tool_not_called_fails(self):
        s = _session("ok")
        s.send("x")
        with pytest.raises(AssertionError, match="should NOT"):
            s.expect_tool_not_called("calc_tariff")

    def test_expect_tool_sequence_passes(self):
        tc = [
            [
                {"name": "search_db"},
                {"name": "calc_tariff"},
                {"name": "format_result"},
            ]
        ]
        s = _session("ok", tool_calls=tc)
        s.send("x").expect_tool_sequence(
            ["search_db", "calc_tariff", "format_result"]
        )

    def test_expect_tool_sequence_wrong_order(self):
        tc = [[{"name": "a"}, {"name": "b"}]]
        s = _session("ok", tool_calls=tc)
        s.send("x")
        with pytest.raises(AssertionError, match="expected tool sequence"):
            s.expect_tool_sequence(["b", "a"])

    def test_expect_tool_params_passes(self):
        tc = [[{"name": "calc", "arguments": {"brand": "BMW", "year": 2024}}]]
        s = _session("ok", tool_calls=tc)
        s.send("x").expect_tool_params("calc", {"brand": "BMW"})

    def test_expect_tool_params_missing_key(self):
        tc = [[{"name": "calc", "arguments": {"brand": "BMW"}}]]
        s = _session("ok", tool_calls=tc)
        s.send("x")
        with pytest.raises(AssertionError, match="missing param"):
            s.expect_tool_params("calc", {"color": "red"})

    def test_expect_tool_params_wrong_value(self):
        tc = [[{"name": "calc", "arguments": {"brand": "BMW"}}]]
        s = _session("ok", tool_calls=tc)
        s.send("x")
        with pytest.raises(AssertionError, match="expected 'Audi'"):
            s.expect_tool_params("calc", {"brand": "Audi"})

    def test_expect_tool_params_tool_not_found(self):
        tc = [[{"name": "calc", "arguments": {}}]]
        s = _session("ok", tool_calls=tc)
        s.send("x")
        with pytest.raises(AssertionError, match="not found"):
            s.expect_tool_params("unknown", {"x": 1})

    def test_expect_tool_params_uses_parameters_key(self):
        tc = [[{"name": "calc", "parameters": {"brand": "BMW"}}]]
        s = _session("ok", tool_calls=tc)
        s.send("x").expect_tool_params("calc", {"brand": "BMW"})

    def test_expect_tool_count_exactly(self):
        tc = [[{"name": "calc"}, {"name": "calc"}, {"name": "log"}]]
        s = _session("ok", tool_calls=tc)
        s.send("x").expect_tool_count("calc", exactly=2)

    def test_expect_tool_count_at_least(self):
        tc = [[{"name": "calc"}, {"name": "calc"}]]
        s = _session("ok", tool_calls=tc)
        s.send("x").expect_tool_count("calc", at_least=1)

    def test_expect_tool_count_at_most(self):
        tc = [[{"name": "calc"}]]
        s = _session("ok", tool_calls=tc)
        s.send("x").expect_tool_count("calc", at_most=2)

    def test_expect_tool_count_exactly_fails(self):
        tc = [[{"name": "calc"}, {"name": "calc"}]]
        s = _session("ok", tool_calls=tc)
        s.send("x")
        with pytest.raises(AssertionError, match="exactly 1"):
            s.expect_tool_count("calc", exactly=1)

    def test_expect_tool_count_zero(self):
        tc = [[{"name": "other"}]]
        s = _session("ok", tool_calls=tc)
        s.send("x").expect_tool_count("calc", exactly=0)

    def test_tool_calls_empty(self):
        tc = [[]]
        s = _session("ok", tool_calls=tc)
        s.send("x")
        with pytest.raises(AssertionError, match="expected tool"):
            s.expect_tool_called("anything")


class TestGroundedness:
    def test_expect_grounded_passes(self):
        s = _session("The price is 5000 rubles for BMW X5")
        s.send("x").expect_grounded(["5000", "bmw"])

    def test_expect_grounded_fails(self):
        s = _session("The price is 5000 rubles")
        s.send("x")
        with pytest.raises(AssertionError, match="missing facts"):
            s.expect_grounded(["5000", "audi"])

    def test_expect_grounded_case_insensitive(self):
        s = _session("Moscow is the capital")
        s.send("x").expect_grounded(["moscow", "CAPITAL"])

    def test_groundedness_metric_registered(self):
        from agent_test_kit.metrics import default_registry
        reg = default_registry()
        assert reg.contains("groundedness")
        assert reg.contains("faithfulness")


class TestSessionReviewFixes:
    def test_run_n_times_uses_real_semantic_scores(self):
        s = _session("Здравствуйте!", judge=_RuleJudge())
        dist = s.run_n_times(
            "Привет",
            n=3,
            evaluate_metric="politeness",
        )
        assert dist.n == 3
        assert all(r.score == pytest.approx(0.75) for r in dist.results)
        assert dist.mean_score == pytest.approx(0.75)

    def test_run_n_times_parallel_requires_client_factory(self):
        s = _session("ok")
        with pytest.raises(ValueError, match="client_factory"):
            s.run_n_times("x", n=2, parallel=True)

    def test_evaluate_invalid_engine_fails_fast(self):
        s = _session("Здравствуйте!", judge=_RuleJudge())
        s.send("Привет")
        with pytest.raises(ValueError, match="Unsupported evaluate engine"):
            s.evaluate("politeness", engine="unknown")

    def test_reset_clears_last_eval_result(self):
        s = _session("Здравствуйте!", judge=_RuleJudge())
        s.send("Привет")
        s.evaluate_direct("politeness")
        assert s.last_eval_result is not None
        s.reset()
        assert s.last_eval_result is None

    def test_evaluate_uses_direct_engine_by_default(self):
        s = _session("Здравствуйте!", judge=_RuleJudge())
        s.send("Привет")
        s.evaluate("politeness", threshold=0.2)
        assert s.last_eval_result is not None
        assert s.last_eval_result.passed


class TestAgentResponseToolCalls:
    def test_tool_calls_property(self):
        r = AgentResponse(
            text="ok", metadata={"tool_calls": [{"name": "a"}]}
        )
        assert r.tool_calls == [{"name": "a"}]

    def test_tool_calls_property_empty(self):
        r = AgentResponse(text="ok")
        assert r.tool_calls == []
