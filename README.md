# agent-test-kit

Переиспользуемый фреймворк для тестирования LLM-агентов.

- **Fluent API** — тесты читаются как пользовательские сценарии
- **Два уровня проверок** — детерминированные `expect_*` + семантические LLM-as-Judge `evaluate()`
- **MetricRegistry** — именованные критерии оценки, built-in + свои
- **90+ готовых тестов** — edge cases, security, format, style, language, scope, memory, latency, concurrency из коробки
- **Единый конфиг** — `agent-test-kit.toml` для всех порогов и параметров
- **pytest plugin** — маркеры `judge` / `slow` + фикстура `atk_config`
- **Опциональный Allure** — автоматические labels, environment и session / G-Eval attachments

## Установка

```bash
pip install -e .

# с конкретным LLM-судьёй:
pip install -e ".[openai]"
pip install -e ".[anthropic]"
pip install -e ".[gigachat]"

# все судьи сразу:
pip install -e ".[all]"

# Allure-репорты:
pip install -e ".[allure]"
```

## Quick Start

### 1. Реализовать клиент

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

### 2. (Опционально) Расширить сессию

```python
from agent_test_kit import AgentSession

class MySession(AgentSession):
    def expect_stays_on_topic(self, terms: set[str]):
        text = self.last_response.text.lower()
        assert any(t in text for t in terms), f"Off-topic: {self.last_text[:200]}"
        return self
```

### 3. Зарегистрировать метрики

```python
from agent_test_kit import default_registry

registry = default_registry()  # включает built-in (prompt_injection_refusal, politeness, …)
registry.register("my_metric", "1. Criterion A\n2. Criterion B")
```

### 4. Настроить conftest.py

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

### 5. Писать тесты

```python
import pytest
from agent_test_kit import run_dialogue

def test_happy_path(session):
    session.send("Привет")
    session.expect_response_ok()
    session.expect_contains("добро пожаловать")

@pytest.mark.judge
def test_polite_response(session_judge):
    session_judge.send("Привет")
    # threshold берётся из [evaluate].default_threshold
    session_judge.evaluate("politeness")

@pytest.mark.slow
def test_long_dialogue(session):
    run_dialogue(session, [
        "Шаг 1", "Шаг 2", "Шаг 3", "Шаг 4", "Шаг 5",
    ])
    assert session.turn >= 5
```

### 6. Запуск

```bash
# быстрый прогон (без LLM-судьи и медленных тестов)
python -m pytest -m "not judge and not slow"

# полный прогон
python -m pytest
```

## Готовые тест-кейсы (Generic Tests)

Фреймворк включает **90+ готовых тестов**, общих для всех LLM-агентов. Чтобы подключить их, достаточно наследовать нужный класс — тесты заработают с вашей фикстурой `session`:

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

| Категория | Класс | Тестов | Маркеры |
|-----------|-------|--------|---------|
| Граничные случаи | `GenericEdgeCaseTests` | 21 | — |
| Prompt security | `GenericPromptSecurityTests` | 9 | `judge` |
| Social engineering | `GenericSocialEngineeringTests` | 2 | `judge` |
| Jailbreak | `GenericJailbreakResistanceTests` | 4 | `judge` |
| Privacy | `GenericPrivacyTests` | 6 | `judge` (1) |
| Payload safety | `GenericPayloadSafetyTests` | 4 | — |
| Поверхностный формат | `GenericSurfaceFormatTests` | 9 | — |
| Стиль общения | `GenericStyleTests` | 4 | `judge` |
| Языковая политика | `GenericLanguageTests` | 2 | — |
| Off-topic отказ | `GenericOffTopicRefusalTests` | 10 | `judge` |
| Mixed intent | `GenericMixedIntentTests` | 2 | `judge` |
| Scope recovery | `GenericScopeRecoveryTests` | 2 | `judge` |
| Recall | `GenericRecallTests` | 4 | `judge` (2) |
| Исправления | `GenericCorrectionTests` | 3 | `judge` (1) |
| Длинный контекст | `GenericLongContextTests` | 3 | `judge`, `slow` |
| Reproducibility | `GenericReproducibilityTests` | 2 | `slow` |
| Paraphrase consistency | `GenericParaphraseConsistencyTests` | 2 | `slow` |
| Session resilience | `GenericSessionResilienceTests` | 2 | `slow` |
| Латентность | `GenericLatencyTests` | 4 | `slow` |
| Параллельность | `GenericConcurrencyTests` | 2 | `slow` |

## Архитектура

```
agent_test_kit/
├── client.py         BaseAgentClient (ABC) — HTTP-клиент
├── response.py       AgentResponse — модель ответа
├── session.py        AgentSession — fluent API + expect_* + evaluate()
├── judge.py          BaseLLMJudge (ABC) + OpenAI / Anthropic / GigaChat
├── metrics.py        MetricRegistry + built-in метрики
├── helpers.py        run_dialogue и утилиты
├── pytest_plugin.py  Маркеры judge/slow (auto-registered)
└── generic_tests/    90+ готовых тест-кейсов по 20 категориям
```

## LLM-судьи

Фреймворк включает три готовых реализации + базовый класс для своих:

### OpenAI (GPT-4o, GPT-4.1, o3 и любой OpenAI-совместимый API)

```python
from agent_test_kit import OpenAIJudge

judge = OpenAIJudge(model_name="gpt-4o")                  # ключ из OPENAI_API_KEY
judge = OpenAIJudge(model_name="gpt-4o", api_key="sk-…")  # ключ явно
judge = OpenAIJudge(                                        # свой endpoint (vLLM, LiteLLM…)
    model_name="my-model",
    base_url="http://localhost:8000/v1",
    api_key="token",
)
```

### Anthropic (Claude)

```python
from agent_test_kit import AnthropicJudge

judge = AnthropicJudge(model_name="claude-sonnet-4-20250514")  # ключ из ANTHROPIC_API_KEY
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

### Подключение в conftest.py

```python
import pytest
from agent_test_kit import OpenAIJudge  # или AnthropicJudge, GigaChatJudge

@pytest.fixture(scope="session")
def judge_llm():
    return OpenAIJudge(model_name="gpt-4o")
```

## Конфигурация

Все пороги и параметры вынесены в один TOML-файл. Положите `agent-test-kit.toml` в корень проекта (рядом с `pyproject.toml`):

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

Альтернативно, секции можно добавить в `pyproject.toml`:

```toml
[tool.agent-test-kit.latency]
first_message_latency = 45.0
```

Порядок загрузки: `agent-test-kit.toml` → `[tool.agent-test-kit]` в `pyproject.toml` → встроенные значения по умолчанию.

Конфиг доступен через фикстуру `atk_config` или программно:

```python
from agent_test_kit import get_config

cfg = get_config()
cfg.latency.first_message_latency  # 45.0
```

Если не передавать `threshold` в `evaluate()`, используется `cfg.evaluate.default_threshold`.

## Allure

Интеграция работает автоматически через pytest-плагин, если установлен extra `allure` и тесты запускаются с `--alluredir`.

```bash
pip install -e ".[dev,allure]"
python -m pytest --alluredir .allure-results
allure serve .allure-results
```

Что добавляется в отчёт автоматически:

- labels `ATK` / feature / markers
- `environment.properties` и `categories.json`
- `test-meta.json`
- trace из `AgentSession`, история диалога и последний G-Eval result

Вложения проходят через базовую санитизацию: ключи вроде `api_key` / `token` редактируются, а типовые email / phone / card-паттерны в тексте маскируются.

## Расширение

- **Свой клиент**: наследуйте `BaseAgentClient`, реализуйте `create_session` и `send_message`
- **Свои проверки**: наследуйте `AgentSession`, добавьте `expect_*` методы
- **Свой LLM-судья**: наследуйте `BaseLLMJudge`, реализуйте `generate` и `get_model_name`
- **Свои метрики**: `registry.register("name", "criteria text")`

### Безопасное логирование HTTP payload

По умолчанию `BaseAgentClient` не логирует тела запросов/ответов (`log_payloads=False`), чтобы не утекали чувствительные данные.

```python
client = MyAgentClient(
    base_url="https://my-agent.example.com",
    log_payloads=True,  # включать только при локальной отладке
)
```

## Built-in метрики

| Метрика | Что оценивает |
|---------|---------------|
| `prompt_injection_refusal` | Отказ на прямой prompt injection |
| `prompt_extraction_refusal` | Отказ на попытку извлечь скрытые инструкции |
| `social_engineering_refusal` | Устойчивость к манипуляции через авторитет и срочность |
| `jailbreak_resistance` | Устойчивость к jailbreak-попыткам |
| `data_isolation_refusal` | Отказ на раскрытие данных других пользователей / сессий |
| `politeness` | Вежливость и стиль общения |
| `context_retention` | Удержание контекста диалога |
| `off_topic_refusal` | Корректный отказ на явно off-topic запрос |
| `mixed_intent_handling` | Обработка смешанного in-scope + off-topic запроса |
| `scope_recovery` | Возврат к основному сценарию после detour |
| `correction_handling` | Обработка исправлений от пользователя |
| `data_extraction` | Корректность извлечения данных |

## Лицензия

MIT
