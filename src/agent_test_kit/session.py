"""
AgentSession — fluent API for testing LLM agents.

Provides a chain-able interface where every ``expect_*`` and ``evaluate``
method returns ``self``, so assertions read like a natural-language spec::

    session.send("Hello").expect_response_ok().expect_contains("welcome")

AgentSession — fluent API для тестирования LLM-агентов. Предоставляет цепочечный
интерфейс, где каждый метод ``expect_*`` и ``evaluate`` возвращает ``self``,
так что утверждения читаются как спецификация на естественном языке.
"""
from __future__ import annotations

import logging
import re
import time
from typing import Any

from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from agent_test_kit.client import BaseAgentClient
from agent_test_kit.config import get_config
from agent_test_kit.judge import BaseLLMJudge
from agent_test_kit.metrics import MetricRegistry, default_registry
from agent_test_kit.response import AgentResponse

logger = logging.getLogger(__name__)


class AgentSession:
    """Fluent API for one conversation with an LLM agent.

    Lifecycle::

        session = AgentSession(client, judge, registry)
        session.init_session(epk_id=111)   # calls client.create_session()
        session.send("Hi")                 # calls client.send_message()
        session.expect_response_ok()       # deterministic assertion
        session.evaluate("politeness")     # LLM-as-Judge assertion

    Subclass to add domain-specific ``expect_*`` methods.

    Fluent API для одного диалога с LLM-агентом. Подкласс можно расширить
    доменными методами ``expect_*``.
    """

    def __init__(
        self,
        client: BaseAgentClient,
        judge: BaseLLMJudge | None = None,
        registry: MetricRegistry | None = None,
    ):
        self._client = client
        self._judge = judge
        self._registry = registry or default_registry()
        self._init_data: dict[str, Any] | None = None
        self._init_message: str = ""
        self._history: list[dict[str, str]] = []
        self._last: AgentResponse | None = None
        self._turn: int = 0
        self._timings: list[float] = []

    # -- lifecycle ----------------------------------------------------------

    def init_session(self, **kwargs: Any) -> "AgentSession":
        """Call ``client.create_session`` and store the init payload. / Вызывает ``client.create_session`` и сохраняет init payload."""
        self._init_data = self._client.create_session(**kwargs)
        self._init_message = self._init_data.get("message", "")
        if self._init_message:
            self._history = [{"role": "assistant", "content": self._init_message}]
        return self

    def reset(self, **kwargs: Any) -> "AgentSession":
        """Discard all state and start a fresh session. / Сбрасывает всё состояние и начинает новую сессию."""
        self._client.reset()
        self._history = []
        self._last = None
        self._turn = 0
        self._timings = []
        self._init_data = None
        self._init_message = ""
        return self.init_session(**kwargs)

    # -- sending messages ---------------------------------------------------

    def send(self, message: str, **kwargs: Any) -> "AgentSession":
        """Send a user message via ``client.send_message``. / Отправляет сообщение пользователя через ``client.send_message``."""
        self._turn += 1
        start = time.perf_counter()
        self._last = self._client.send_message(message, **kwargs)
        elapsed = time.perf_counter() - start
        self._timings.append(elapsed)
        self._history.append({"role": "user", "content": message})
        self._history.append({"role": "assistant", "content": self._last.text})
        return self

    # ======================================================================
    # Deterministic assertions (expect_*)
    # ======================================================================

    # -- response status ----------------------------------------------------

    def expect_response_ok(self) -> "AgentSession":
        """Assert HTTP 200 and non-empty text. / Проверяет HTTP 200 и непустой текст ответа."""
        self._need_response()
        assert self._last.status_code == 200, (
            f"Turn {self._turn}: HTTP {self._last.status_code}"
        )
        assert self._last.text.strip(), (
            f"Turn {self._turn}: response text is empty. Raw: {self._last.raw}"
        )
        return self

    # -- keyword checks -----------------------------------------------------

    def expect_contains(self, *keywords: str) -> "AgentSession":
        """Response contains ALL keywords (case-insensitive). / Ответ содержит ВСЕ ключевые слова (без учёта регистра)."""
        self._need_response()
        text = self._last.text.lower()
        missing = [kw for kw in keywords if kw.lower() not in text]
        assert not missing, (
            f"Turn {self._turn}: response missing keywords: {missing}\n"
            f"Response: '{self._last.text[:300]}'"
        )
        return self

    def expect_contains_any(self, *keywords: str) -> "AgentSession":
        """Response contains AT LEAST ONE keyword. / Ответ содержит ХОТЯ БЫ ОДНО ключевое слово."""
        self._need_response()
        text = self._last.text.lower()
        found = [kw for kw in keywords if kw.lower() in text]
        assert found, (
            f"Turn {self._turn}: response contains none of: {list(keywords)}\n"
            f"Response: '{self._last.text[:300]}'"
        )
        return self

    def expect_not_contains(self, *keywords: str) -> "AgentSession":
        """Response does NOT contain any of the keywords. / Ответ НЕ содержит ни одного из ключевых слов."""
        self._need_response()
        text = self._last.text.lower()
        found = [kw for kw in keywords if kw.lower() in text]
        assert not found, (
            f"Turn {self._turn}: response contains forbidden: {found}\n"
            f"Response: '{self._last.text[:300]}'"
        )
        return self

    # -- latency & length ---------------------------------------------------

    def expect_latency_under(self, seconds: float) -> "AgentSession":
        """Last response arrived within *seconds*. / Последний ответ получен за время менее *seconds* секунд."""
        assert self._timings, "No timings recorded. Call send() first."
        actual = self._timings[-1]
        assert actual < seconds, (
            f"Turn {self._turn}: latency {actual:.2f}s > threshold {seconds}s"
        )
        return self

    def expect_response_length(
        self, min_chars: int = 1, max_chars: int = 5000,
    ) -> "AgentSession":
        """Response length is within ``[min_chars, max_chars]``. / Длина ответа в пределах ``[min_chars, max_chars]``."""
        self._need_response()
        length = len(self._last.text)
        assert min_chars <= length <= max_chars, (
            f"Turn {self._turn}: length {length}, expected [{min_chars}..{max_chars}]"
        )
        return self

    # -- format & style -----------------------------------------------------

    def expect_asks_question(self) -> "AgentSession":
        """Response ends with a question mark. / Ответ заканчивается вопросительным знаком."""
        self._need_response()
        assert self._last.text.rstrip().endswith("?"), (
            f"Turn {self._turn}: response does not end with '?'\n"
            f"Response: '{self._last.text[:300]}'"
        )
        return self

    def expect_formal_you(self) -> "AgentSession":
        """Russian-language check: formal 'Вы', no informal 'ты/тебе/тебя'. / Проверка русского языка: вежливое «Вы», без неформального «ты/тебе/тебя»."""
        self._need_response()
        informal = re.findall(
            r"\b(ты|тебе|тебя|тебой|твой|твоя|твоё|твои|твоего|твоей|твоим)\b",
            self._last.text,
            re.IGNORECASE,
        )
        assert not informal, (
            f"Turn {self._turn}: informal pronouns found: {informal}\n"
            f"Response: '{self._last.text[:300]}'"
        )
        return self

    def expect_no_pii(
        self, patterns: list[str] | None = None,
    ) -> "AgentSession":
        """Response does not contain PII matching *patterns* (regex).

        When *patterns* is ``None``, uses ``security.pii_patterns`` from
        ``agent-test-kit.toml`` (defaults to passport, SSN, credit card,
        phone number patterns).

        Ответ не содержит PII, соответствующего *patterns* (regex). Если
        *patterns* — ``None``, используются ``security.pii_patterns`` из
        ``agent-test-kit.toml`` (по умолчанию: паспорт, SSN, карта, телефон).
        """
        self._need_response()
        if patterns is None:
            patterns = get_config().security.pii_patterns
        for pat in patterns:
            matches = re.findall(pat, self._last.text)
            assert not matches, (
                f"Turn {self._turn}: PII detected (pattern '{pat}'): {matches}"
            )
        return self

    # -- metadata (node, tool_calls, …) ------------------------------------

    def expect_metadata(self, key: str, expected: Any = None) -> "AgentSession":
        """Assert that ``metadata[key]`` exists (and optionally equals *expected*). / Проверяет наличие ``metadata[key]`` (и опционально равенство *expected*)."""
        self._need_response()
        assert key in self._last.metadata, (
            f"Turn {self._turn}: metadata key '{key}' missing. "
            f"Keys: {list(self._last.metadata.keys())}"
        )
        if expected is not None:
            assert self._last.metadata[key] == expected, (
                f"Turn {self._turn}: metadata['{key}'] = "
                f"'{self._last.metadata[key]}', expected '{expected}'"
            )
        return self

    def expect_raw_field(self, field: str, expected: Any = None) -> "AgentSession":
        """Assert presence (and optional value) of a field in raw JSON. / Проверяет наличие поля (и опционально значение) в сыром JSON."""
        self._need_response()
        assert field in self._last.raw, (
            f"Turn {self._turn}: raw field '{field}' missing. "
            f"Keys: {list(self._last.raw.keys())}"
        )
        if expected is not None:
            assert self._last.raw[field] == expected, (
                f"Turn {self._turn}: raw['{field}'] = "
                f"'{self._last.raw[field]}', expected '{expected}'"
            )
        return self

    # -- session liveness ---------------------------------------------------

    def expect_session_alive(self) -> "AgentSession":
        """Session is still active: send a probe message and check 200. / Сессия активна: отправляет пробное сообщение и проверяет 200."""
        assert self._client.session_id, "session_id is not set"
        self.send("test")
        self.expect_response_ok()
        return self

    # -- init-response checks -----------------------------------------------

    def expect_init_contains(self, *keywords: str) -> "AgentSession":
        """Init message contains ALL keywords (case-insensitive). / Init-сообщение содержит ВСЕ ключевые слова (без учёта регистра)."""
        assert self._init_message, "No init message. Call init_session() first."
        text = self._init_message.lower()
        missing = [kw for kw in keywords if kw.lower() not in text]
        assert not missing, (
            f"Init message missing keywords: {missing}\n"
            f"Init message: '{self._init_message[:300]}'"
        )
        return self

    def expect_init_data(self, key: str, expected: Any = None) -> "AgentSession":
        """Assert that init payload contains *key* (and optionally equals *expected*). / Проверяет, что init payload содержит *key* (и опционально равен *expected*)."""
        assert self._init_data is not None, "No init data. Call init_session() first."
        assert key in self._init_data, (
            f"Init data missing key '{key}'. Keys: {list(self._init_data.keys())}"
        )
        if expected is not None:
            assert self._init_data[key] == expected, (
                f"init_data['{key}'] = '{self._init_data[key]}', expected '{expected}'"
            )
        return self

    # ======================================================================
    # LLM-as-Judge (DeepEval GEval)
    # ======================================================================

    def evaluate(
        self,
        metric_name: str,
        threshold: float | None = None,
        criteria: str | None = None,
    ) -> "AgentSession":
        """Run an LLM-as-Judge evaluation on the last response.

        Looks up *metric_name* in the registry unless *criteria* is given
        explicitly.

        Запускает LLM-as-Judge оценку последнего ответа. Ищет *metric_name*
        в реестре, если *criteria* не задан явно.
        """
        assert self._judge is not None, (
            "No LLM judge configured. Pass a BaseLLMJudge to AgentSession."
        )

        if criteria is None:
            criteria = self._registry.get(metric_name)
        if threshold is None:
            threshold = get_config().evaluate.default_threshold

        if self._last is not None:
            actual_output = self._last.text
            last_input = (
                self._history[-2]["content"] if len(self._history) >= 2 else ""
            )
        elif self._init_message:
            actual_output = self._init_message
            last_input = "session init"
        else:
            raise AssertionError("No response available. Call init_session() or send().")

        logger.debug(
            "evaluate(%s, threshold=%.2f)\n  input:  %s\n  output: %s",
            metric_name, threshold, last_input[:300], actual_output[:500],
        )

        metric = GEval(
            name=metric_name,
            model=self._judge,
            criteria=criteria,
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
            ],
            threshold=threshold,
        )
        test_case = LLMTestCase(input=last_input, actual_output=actual_output)
        assert_test(test_case, [metric])
        return self

    # ======================================================================
    # Public properties
    # ======================================================================

    @property
    def init_data(self) -> dict[str, Any]:
        assert self._init_data is not None, "No init data. Call init_session() first."
        return self._init_data

    @property
    def init_message(self) -> str:
        return self._init_message

    @property
    def last_response(self) -> AgentResponse:
        self._need_response()
        return self._last  # type: ignore[return-value]

    @property
    def last_text(self) -> str:
        self._need_response()
        return self._last.text  # type: ignore[union-attr]

    @property
    def history(self) -> list[dict[str, str]]:
        return self._history

    @property
    def turn(self) -> int:
        return self._turn

    @property
    def timings(self) -> list[float]:
        return self._timings

    @property
    def registry(self) -> MetricRegistry:
        return self._registry

    # ======================================================================
    # Internal
    # ======================================================================

    def _need_response(self) -> None:
        """Ensure a response exists; raise if not. / Проверяет наличие ответа; вызывает исключение при отсутствии."""
        assert self._last is not None, (
            "No response from agent. Call send() first."
        )
