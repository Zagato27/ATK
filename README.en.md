# agent-test-kit

A reusable testing framework for LLM agents.

- **Fluent API** — tests read like user scenarios
- **Two levels of checks** — deterministic `expect_*` + semantic LLM-as-Judge `evaluate()`
- **MetricRegistry** — named evaluation criteria, built-in + custom
- **90+ ready-made tests** — edge cases, security, format, style, language, scope, memory, latency, concurrency out of the box
- **Single config file** — `agent-test-kit.toml` for all thresholds and parameters
- **pytest plugin** — `judge` / `slow` markers + `atk_config` fixture
- **Optional Allure integration** — automatic labels, environment data, and session / G-Eval attachments

## Installation

```bash
pip install -e .

# with a specific LLM judge:
pip install -e ".[openai]"
pip install -e ".[anthropic]"
pip install -e ".[gigachat]"

# all judges at once:
pip install -e ".[all]"

# Allure reports:
pip install -e ".[allure]"
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

registry = default_registry()  # includes built-in (prompt_injection_refusal, politeness, …)
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

The framework ships with **90+ ready-made tests** common to all LLM agents. To use them, simply inherit the desired class — tests will run automatically with your `session` fixture:

```python
# tests/test_generic.py
from agent_test_kit.generic_tests import (
    GenericEdgeCaseTests,
    GenericPromptSecurityTests,
    GenericSocialEngineeringTests,
    GenericJailbreakResistanceTests,
    GenericPrivacyTests,
    GenericPayloadSafetyTests,
    GenericSurfaceFormatTests,
    GenericStyleTests,
    GenericLanguageTests,
    GenericOffTopicRefusalTests,
    GenericMixedIntentTests,
    GenericScopeRecoveryTests,
    GenericRecallTests,
    GenericCorrectionTests,
    GenericLongContextTests,
    GenericReproducibilityTests,
    GenericParaphraseConsistencyTests,
    GenericSessionResilienceTests,
    GenericLatencyTests,
    GenericConcurrencyTests,
)

class TestEdgeCases(GenericEdgeCaseTests):
    pass

class TestPromptSecurity(GenericPromptSecurityTests):
    pass

class TestSocialEngineering(GenericSocialEngineeringTests):
    pass

class TestJailbreak(GenericJailbreakResistanceTests):
    pass

class TestPrivacy(GenericPrivacyTests):
    pass

class TestPayloadSafety(GenericPayloadSafetyTests):
    pass

class TestSurfaceFormat(GenericSurfaceFormatTests):
    pass

class TestStyle(GenericStyleTests):
    pass

class TestLanguage(GenericLanguageTests):
    pass

class TestOffTopic(GenericOffTopicRefusalTests):
    pass

class TestMixedIntent(GenericMixedIntentTests):
    pass

class TestScopeRecovery(GenericScopeRecoveryTests):
    pass

class TestRecall(GenericRecallTests):
    pass

class TestCorrections(GenericCorrectionTests):
    pass

class TestLongContext(GenericLongContextTests):
    pass

class TestReproducibility(GenericReproducibilityTests):
    pass

class TestParaphraseConsistency(GenericParaphraseConsistencyTests):
    pass

class TestSessionResilience(GenericSessionResilienceTests):
    pass

class TestLatency(GenericLatencyTests):
    pass

class TestConcurrency(GenericConcurrencyTests):
    pass
```

| Category | Class | Tests | Markers |
|----------|-------|-------|---------|
| Edge cases | `GenericEdgeCaseTests` | 21 | — |
| Prompt security | `GenericPromptSecurityTests` | 9 | `judge` |
| Social engineering | `GenericSocialEngineeringTests` | 2 | `judge` |
| Jailbreak | `GenericJailbreakResistanceTests` | 4 | `judge` |
| Privacy | `GenericPrivacyTests` | 6 | `judge` (1) |
| Payload safety | `GenericPayloadSafetyTests` | 4 | — |
| Surface format | `GenericSurfaceFormatTests` | 9 | — |
| Communication style | `GenericStyleTests` | 4 | `judge` |
| Language policy | `GenericLanguageTests` | 2 | — |
| Off-topic refusal | `GenericOffTopicRefusalTests` | 10 | `judge` |
| Mixed intent | `GenericMixedIntentTests` | 2 | `judge` |
| Scope recovery | `GenericScopeRecoveryTests` | 2 | `judge` |
| Recall | `GenericRecallTests` | 4 | `judge` (2) |
| Corrections | `GenericCorrectionTests` | 3 | `judge` (1) |
| Long context | `GenericLongContextTests` | 3 | `judge`, `slow` |
| Reproducibility | `GenericReproducibilityTests` | 2 | `slow` |
| Paraphrase consistency | `GenericParaphraseConsistencyTests` | 2 | `slow` |
| Session resilience | `GenericSessionResilienceTests` | 2 | `slow` |
| Latency | `GenericLatencyTests` | 4 | `slow` |
| Concurrency | `GenericConcurrencyTests` | 2 | `slow` |

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
└── generic_tests/    90+ ready-made test cases across 20 categories
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
[latency]
first_message_latency = 45.0

[concurrency]
parallel_count = 5

[prompt_security]
judge_threshold = 0.8

[privacy]
judge_threshold = 0.7
pii_patterns = [
    "\\b\\d{4}\\s?\\d{6}\\b",
    "\\b\\d{3}-\\d{2}-\\d{4}\\b",
    "[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\\.[a-zA-Z]{2,}",
]

[recall]
judge_threshold = 0.8

[corrections]
judge_threshold = 0.7

[long_context]
judge_threshold = 0.6

[evaluate]
default_threshold = 0.7
```

Alternatively, add sections under `[tool.agent-test-kit]` in `pyproject.toml`:

```toml
[tool.agent-test-kit.latency]
first_message_latency = 45.0
```

Loading order: `agent-test-kit.toml` → `[tool.agent-test-kit]` in `pyproject.toml` → built-in defaults.

The config is available via the `atk_config` fixture or programmatically:

```python
from agent_test_kit import get_config

cfg = get_config()
cfg.latency.first_message_latency  # 45.0
```

If you omit `threshold` in `evaluate()`, it uses `cfg.evaluate.default_threshold`.

## Allure

The integration works automatically through the pytest plugin when the `allure` extra is installed and tests are run with `--alluredir`.

```bash
pip install -e ".[dev,allure]"
python -m pytest --alluredir .allure-results
allure serve .allure-results
```

The report is enriched automatically with:

- `ATK` / feature / marker labels
- `environment.properties` and `categories.json`
- `test-meta.json`
- `AgentSession` traces, dialogue history, and the latest G-Eval result

Attachments go through basic sanitization: keys like `api_key` / `token` are redacted, and common email / phone / card patterns in text are masked.

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
| `prompt_injection_refusal` | Refusal of direct prompt injection |
| `prompt_extraction_refusal` | Refusal to reveal hidden instructions |
| `social_engineering_refusal` | Resistance to authority- / urgency-based manipulation |
| `jailbreak_resistance` | Resistance to jailbreak attempts |
| `data_isolation_refusal` | Refusal to reveal other users' / sessions' data |
| `politeness` | Politeness and communication style |
| `context_retention` | Maintaining dialogue context |
| `off_topic_refusal` | Refusing clearly off-topic requests |
| `mixed_intent_handling` | Handling mixed in-scope + off-topic requests |
| `scope_recovery` | Returning to the main scenario after a detour |
| `correction_handling` | Handling user corrections |
| `data_extraction` | Correctness of data extraction |

## License

MIT
