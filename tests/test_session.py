"""
Unit tests for AgentSession deterministic assertions.

Uses MockClient — no network, no LLM judge.
"""
import pytest

from agent_test_kit import AgentResponse, AgentSession, BaseAgentClient
from agent_test_kit.helpers import run_dialogue
from agent_test_kit.metrics import MetricRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _Client(BaseAgentClient):
    """Tiny mock that returns one fixed response per send."""

    def __init__(self, replies: list[str]):
        super().__init__(base_url="http://test", timeout=1, verify=False)
        self._replies = replies
        self._idx = 0

    def create_session(self, **kw):
        self.session_id = "s1"
        return {"session_id": "s1", "message": "Welcome!"}

    def send_message(self, message, **kw):
        text = self._replies[self._idx % len(self._replies)]
        self._idx += 1
        return AgentResponse(
            text=text, status_code=200,
            metadata={"node": "main", "tool_calls": [{"name": "calc_tariff"}]},
            raw={"response": text, "node": "main"},
        )

    def reset(self):
        super().reset()
        self._idx = 0


def _session(*replies: str) -> AgentSession:
    client = _Client(list(replies))
    s = AgentSession(client=client)
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
