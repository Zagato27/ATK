# agent-test-kit

A reusable testing framework for LLM agents.

- **Fluent API** — tests read like user scenarios
- **Two levels of checks** — deterministic `expect_*` + semantic LLM-as-Judge `evaluate()`
- **MetricRegistry** — named evaluation criteria, built-in + custom
- **70+ ready-made tests** — edge cases, security, format, memory, performance out of the box
- **Single config file** — `agent-test-kit.toml` for all thresholds and parameters
- **pytest plugin** — `judge` / `slow` markers + `atk_config` fixture

## Installation

```bash
pip install -e .

# with a specific LLM judge:
pip install -e ".[openai]"
pip install -e ".[anthropic]"
pip install -e ".[gigachat]"

# all judges at once:
pip install -e ".[all]"
```

## Quick Start

### 1. Implement a Client

```python
from agent_test_kit import BaseAgentClient, AgentResponse

class MyAgentClient(BaseAgentClient):
    def create_session(self, **kwargs) -> dict:
        resp = self._post("/api/session/init", {"user_id": kwargs["user_id"]})
        resp.raise_for_status()
        data = resp.json()
        self.session_id = data["session_id"]
        return data

    def send_message(self, message: str, **kwargs) -> AgentResponse:
        resp = self._post("/api/chat", {
            "session_id": self.session_id,
            "message": message,
        })
        resp.raise_for_status()
        return AgentResponse.from_raw(resp.json(), status_code=resp.status_code)
```

### 2. (Optional) Extend the Session

```python
from agent_test_kit import AgentSession

class MySession(AgentSession):
    def expect_stays_on_topic(self, terms: set[str]):
        text = self.last_response.text.lower()
        assert any(t in text for t in terms), f"Off-topic: {self.last_text[:200]}"
        return self
```

### 3. Register Metrics

```python
from agent_test_kit import default_registry

registry = default_registry()  # includes built-in (security_refusal, politeness, …)
registry.register("my_metric", "1. Criterion A\n2. Criterion B")
```

### 4. Configure conftest.py

```python
import pytest
from agent_test_kit import AgentSession, run_dialogue
from my_project.client import MyAgentClient

@pytest.fixture(scope="session")
def agent_client():
    return MyAgentClient(base_url="https://my-agent.example.com")

@pytest.fixture
def session(agent_client):
    s = AgentSession(client=agent_client)
    s.reset(user_id=42)
    return s

@pytest.fixture
def session_judge(agent_client, judge_llm):
    s = AgentSession(client=agent_client, judge=judge_llm)
    s.reset(user_id=42)
    return s
```

### 5. Write Tests

```python
import pytest
from agent_test_kit import run_dialogue

def test_happy_path(session):
    session.send("Hello")
    session.expect_response_ok()
    session.expect_contains("welcome")

@pytest.mark.judge
def test_polite_response(session_judge):
    session_judge.send("Hello")
    # uses [evaluate].default_threshold from config
    session_judge.evaluate("politeness")

@pytest.mark.slow
def test_long_dialogue(session):
    run_dialogue(session, [
        "Step 1", "Step 2", "Step 3", "Step 4", "Step 5",
    ])
    assert session.turn >= 5
```

### 6. Run

```bash
# fast run (no LLM judge, no slow tests)
python -m pytest -m "not judge and not slow"

# full run
python -m pytest
```

## Ready-Made Test Suites (Generic Tests)

The framework ships with **70+ ready-made tests** common to all LLM agents. To use them, simply inherit the desired class — tests will run automatically with your `session` fixture:

```python
# tests/test_generic.py
from agent_test_kit.generic_tests import (
    GenericEdgeCaseTests,
    GenericSecurityTests,
    GenericFormatTests,
    GenericOutOfScopeTests,
    GenericMemoryTests,
    GenericStabilityTests,
    GenericPerformanceTests,
)

class TestEdgeCases(GenericEdgeCaseTests):
    pass

class TestSecurity(GenericSecurityTests):
    pass

class TestFormat(GenericFormatTests):
    pass

class TestOutOfScope(GenericOutOfScopeTests):
    pass

class TestMemory(GenericMemoryTests):
    pass

class TestStability(GenericStabilityTests):
    pass

class TestPerformance(GenericPerformanceTests):
    # override thresholds to match your SLA
    FIRST_MESSAGE_LATENCY = 45.0
    PARALLEL_COUNT = 5
```

| Category | Class | Tests | Markers |
|----------|-------|-------|---------|
| Edge cases | `GenericEdgeCaseTests` | 21 | — |
| Security | `GenericSecurityTests` | 20 | `judge` |
| Format & style | `GenericFormatTests` | 11 | `judge` (2) |
| Out-of-scope | `GenericOutOfScopeTests` | 5 | `judge` |
| Memory & context | `GenericMemoryTests` | 10 | `judge`, `slow` |
| Stability | `GenericStabilityTests` | 4 | `slow`, `judge` |
| Performance | `GenericPerformanceTests` | 7 | `slow` |

## Architecture

```
agent_test_kit/
├── client.py         BaseAgentClient (ABC) — HTTP client
├── response.py       AgentResponse — response model
├── session.py        AgentSession — fluent API + expect_* + evaluate()
├── judge.py          BaseLLMJudge (ABC) + OpenAI / Anthropic / GigaChat
├── metrics.py        MetricRegistry + built-in metrics
├── helpers.py        run_dialogue and utilities
├── pytest_plugin.py  judge/slow markers (auto-registered)
└── generic_tests/    70+ ready-made test cases across 7 categories
```

## LLM Judges

The framework ships with three ready-made implementations plus a base class for custom ones:

### OpenAI (GPT-4o, GPT-4.1, o3, and any OpenAI-compatible API)

```python
from agent_test_kit import OpenAIJudge

judge = OpenAIJudge(model_name="gpt-4o")                  # key from OPENAI_API_KEY
judge = OpenAIJudge(model_name="gpt-4o", api_key="sk-…")  # explicit key
judge = OpenAIJudge(                                        # custom endpoint (vLLM, LiteLLM…)
    model_name="my-model",
    base_url="http://localhost:8000/v1",
    api_key="token",
)
```

### Anthropic (Claude)

```python
from agent_test_kit import AnthropicJudge

judge = AnthropicJudge(model_name="claude-sonnet-4-20250514")  # key from ANTHROPIC_API_KEY
judge = AnthropicJudge(
    model_name="claude-sonnet-4-20250514",
    api_key="sk-ant-…",
    max_tokens=8192,
)
```

### GigaChat (mTLS)

```python
from agent_test_kit import GigaChatJudge

judge = GigaChatJudge(
    model_name="GigaChat-Max",
    base_url="https://gigachat.example.com/api/v1",
    cert_file="certs/client.pem",
    key_file="certs/client.key",
)
```

### Wiring in conftest.py

```python
import pytest
from agent_test_kit import OpenAIJudge  # or AnthropicJudge, GigaChatJudge

@pytest.fixture(scope="session")
def judge_llm():
    return OpenAIJudge(model_name="gpt-4o")
```

## Configuration

All thresholds and parameters live in a single TOML file. Place `agent-test-kit.toml` in your project root (next to `pyproject.toml`):

```toml
[performance]
first_message_latency = 45.0
parallel_count = 5

[security]
judge_threshold = 0.8
pii_patterns = [
    "\\b\\d{4}\\s?\\d{6}\\b",
    "\\b\\d{3}-\\d{2}-\\d{4}\\b",
]

[memory]
context_threshold = 0.8
correction_threshold = 0.7

[evaluate]
default_threshold = 0.7
```

Alternatively, add sections under `[tool.agent-test-kit]` in `pyproject.toml`:

```toml
[tool.agent-test-kit.performance]
first_message_latency = 45.0
```

Loading order: `agent-test-kit.toml` → `[tool.agent-test-kit]` in `pyproject.toml` → built-in defaults.

The config is available via the `atk_config` fixture or programmatically:

```python
from agent_test_kit import get_config

cfg = get_config()
cfg.performance.first_message_latency  # 45.0
```

If you omit `threshold` in `evaluate()`, it uses `cfg.evaluate.default_threshold`.

## Extending

- **Custom client**: subclass `BaseAgentClient`, implement `create_session` and `send_message`
- **Custom assertions**: subclass `AgentSession`, add `expect_*` methods
- **Custom LLM judge**: subclass `BaseLLMJudge`, implement `generate` and `get_model_name`
- **Custom metrics**: `registry.register("name", "criteria text")`

### Safe HTTP Payload Logging

By default, `BaseAgentClient` does not log request/response bodies (`log_payloads=False`) to reduce the chance of leaking sensitive data.

```python
client = MyAgentClient(
    base_url="https://my-agent.example.com",
    log_payloads=True,  # enable only during local debugging
)
```

## Built-in Metrics

| Metric | What it evaluates |
|--------|-------------------|
| `security_refusal` | Refusal of prompt injection / manipulation |
| `politeness` | Politeness and communication style |
| `context_retention` | Maintaining dialogue context |
| `out_of_scope_handling` | Handling off-topic messages |
| `correction_handling` | Handling user corrections |
| `data_extraction` | Correctness of data extraction |

## License

MIT
