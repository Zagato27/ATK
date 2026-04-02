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
from typing import Any, Callable

from agent_test_kit.client import BaseAgentClient
from agent_test_kit.config import get_config
from agent_test_kit.judge import BaseLLMJudge
from agent_test_kit.metrics import MetricRegistry, default_registry
from agent_test_kit.response import AgentResponse
from agent_test_kit.statistical import Distribution, RunResult, run_n_times

logger = logging.getLogger(__name__)

_SESSION_OBSERVER: Callable[["AgentSession"], None] | None = None


def _install_session_observer(
    observer: Callable[["AgentSession"], None] | None,
) -> Callable[["AgentSession"], None] | None:
    global _SESSION_OBSERVER
    previous = _SESSION_OBSERVER
    _SESSION_OBSERVER = observer
    return previous


def _restore_session_observer(
    observer: Callable[["AgentSession"], None] | None,
) -> None:
    global _SESSION_OBSERVER
    _SESSION_OBSERVER = observer


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
        self._last_eval_result: Any = None
        if _SESSION_OBSERVER is not None:
            try:
                _SESSION_OBSERVER(self)
            except Exception:  # pragma: no cover - diagnostics must stay best-effort
                logger.debug("session observer failed", exc_info=True)

    # -- lifecycle ----------------------------------------------------------

    def init_session(self, **kwargs: Any) -> "AgentSession":
        """Call ``client.create_session`` and store the init payload. / Вызывает ``client.create_session`` и сохраняет init payload."""
        if self._should_log_debug():
            logger.debug(
                "session.init_session(): kwargs=%s",
                self._short_repr(kwargs, self._log_prompt_limit()),
            )
        self._init_data = self._client.create_session(**kwargs)
        self._init_message = self._init_data.get("message", "")
        self._last_eval_result = None
        if self._init_message:
            self._history = [{"role": "assistant", "content": self._init_message}]
        if self._should_log_debug():
            logger.debug(
                "session.init_session(): session_id=%s keys=%s",
                self._client.session_id,
                list(self._init_data.keys()),
            )
            if self._init_message:
                logger.debug(
                    "session.init_session(): received init message:\n%s",
                    self._short_text(self._init_message, self._log_response_limit()),
                )
            logger.debug(
                "session.init_session(): raw init payload:\n%s",
                self._short_repr(self._init_data, self._log_response_limit()),
            )
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
        self._last_eval_result = None
        return self.init_session(**kwargs)

    # -- sending messages ---------------------------------------------------

    def send(self, message: str, **kwargs: Any) -> "AgentSession":
        """Send a user message via ``client.send_message``. / Отправляет сообщение пользователя через ``client.send_message``."""
        self._turn += 1
        if self._should_log_debug():
            logger.debug(
                "session.send(turn=%d): outgoing user message:\n%s",
                self._turn,
                self._short_text(message, self._log_prompt_limit()),
            )
            if kwargs:
                logger.debug(
                    "session.send(turn=%d): send kwargs=%s",
                    self._turn,
                    self._short_repr(kwargs, self._log_prompt_limit()),
                )
        start = time.perf_counter()
        self._last = self._client.send_message(message, **kwargs)
        self._last_eval_result = None
        elapsed = time.perf_counter() - start
        self._timings.append(elapsed)
        self._history.append({"role": "user", "content": message})
        self._history.append({"role": "assistant", "content": self._last.text})
        if self._should_log_debug():
            logger.debug(
                "session.send(turn=%d): received response status=%s latency=%.2fs",
                self._turn,
                self._last.status_code,
                elapsed,
            )
            logger.debug(
                "session.send(turn=%d): agent response:\n%s",
                self._turn,
                self._short_text(self._last.text, self._log_response_limit()),
            )
            if self._last.metadata:
                logger.debug(
                    "session.send(turn=%d): response metadata=%s",
                    self._turn,
                    self._short_repr(self._last.metadata, self._log_response_limit()),
                )
            if self._last.raw:
                logger.debug(
                    "session.send(turn=%d): raw response payload=%s",
                    self._turn,
                    self._short_repr(self._last.raw, self._log_response_limit()),
                )
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

        When *patterns* is ``None``, uses ``privacy.pii_patterns`` from
        ``agent-test-kit.toml`` (defaults to passport, SSN, credit card,
        phone number, and email patterns).

        Ответ не содержит PII, соответствующего *patterns* (regex). Если
        *patterns* — ``None``, используются ``privacy.pii_patterns`` из
        ``agent-test-kit.toml`` (по умолчанию: паспорт, SSN, карта, телефон
        и email).
        """
        self._need_response()
        if patterns is None:
            patterns = get_config().privacy.pii_patterns
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

    # -- tool call checks ---------------------------------------------------

    def expect_tool_called(self, name: str) -> "AgentSession":
        """Assert that a tool with *name* was called (case-insensitive substring match). / Проверяет, что тул *name* был вызван."""
        self._need_response()
        names = [tc.get("name", "") for tc in self._last.tool_calls]
        assert any(name.lower() in n.lower() for n in names), (
            f"Turn {self._turn}: expected tool '{name}', "
            f"called: {names}"
        )
        return self

    def expect_tool_not_called(self, *names: str) -> "AgentSession":
        """Assert that none of *names* appear in tool calls. / Проверяет, что ни один из *names* не был вызван."""
        self._need_response()
        called = [tc.get("name", "") for tc in self._last.tool_calls]
        found = [
            n for n in names
            if any(n.lower() in c.lower() for c in called)
        ]
        assert not found, (
            f"Turn {self._turn}: tools should NOT have been called: {found}, "
            f"but called: {called}"
        )
        return self

    def expect_tool_sequence(self, expected: list[str]) -> "AgentSession":
        """Assert that tools were called in exactly this order (case-insensitive). / Проверяет порядок вызова тулов."""
        self._need_response()
        actual = [tc.get("name", "") for tc in self._last.tool_calls]
        actual_lower = [n.lower() for n in actual]
        expected_lower = [n.lower() for n in expected]
        assert actual_lower == expected_lower, (
            f"Turn {self._turn}: expected tool sequence {expected}, "
            f"got {actual}"
        )
        return self

    def expect_tool_params(self, name: str, expected_params: dict) -> "AgentSession":
        """Assert that the call to *name* includes *expected_params* (subset match). / Проверяет параметры вызова тула *name*."""
        self._need_response()
        for tc in self._last.tool_calls:
            tc_name = tc.get("name", "")
            if name.lower() not in tc_name.lower():
                continue
            args = tc.get("arguments", tc.get("parameters", {}))
            for key, value in expected_params.items():
                assert key in args, (
                    f"Turn {self._turn}: tool '{tc_name}' missing param '{key}'. "
                    f"Params: {args}"
                )
                assert args[key] == value, (
                    f"Turn {self._turn}: tool '{tc_name}' param '{key}' = "
                    f"{args[key]!r}, expected {value!r}"
                )
            return self
        called = [tc.get("name", "") for tc in self._last.tool_calls]
        raise AssertionError(
            f"Turn {self._turn}: tool '{name}' not found in calls: {called}"
        )

    def expect_tool_count(
        self,
        name: str,
        *,
        exactly: int | None = None,
        at_least: int | None = None,
        at_most: int | None = None,
    ) -> "AgentSession":
        """Assert tool *name* call count (case-insensitive). / Проверяет количество вызовов тула *name*."""
        self._need_response()
        count = sum(
            1 for tc in self._last.tool_calls
            if name.lower() in tc.get("name", "").lower()
        )
        if exactly is not None:
            assert count == exactly, (
                f"Turn {self._turn}: tool '{name}' called {count} times, "
                f"expected exactly {exactly}"
            )
        if at_least is not None:
            assert count >= at_least, (
                f"Turn {self._turn}: tool '{name}' called {count} times, "
                f"expected at least {at_least}"
            )
        if at_most is not None:
            assert count <= at_most, (
                f"Turn {self._turn}: tool '{name}' called {count} times, "
                f"expected at most {at_most}"
            )
        return self

    # -- groundedness checks ------------------------------------------------

    def expect_grounded(self, facts: list[str]) -> "AgentSession":
        """Assert that every fact appears in the response (case-insensitive).

        Deterministic check: each string from *facts* must be present in
        the agent's last response text. For semantic groundedness use
        ``evaluate("groundedness", context=[...])``.

        Детерминированная проверка: каждый факт из *facts* присутствует в ответе
        агента (без учёта регистра).
        """
        self._need_response()
        text = self._last.text.lower()
        missing = [f for f in facts if f.lower() not in text]
        assert not missing, (
            f"Turn {self._turn}: response not grounded — missing facts: {missing}\n"
            f"Response: '{self._last.text[:300]}'"
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
    # Statistical N-run testing
    # ======================================================================

    def run_n_times(
        self,
        message: str,
        n: int | None = None,
        *,
        evaluate_metric: str | None = None,
        parallel: bool = False,
        init_kwargs: dict[str, Any] | None = None,
        client_factory: Callable[[], BaseAgentClient] | None = None,
    ) -> Distribution:
        """Send *message* N times in fresh sessions and collect a :class:`Distribution`.

        Each run creates a new session via ``init_session(**init_kwargs)``,
        sends *message*, and records pass/fail + latency. If *evaluate_metric*
        is given, runs ``evaluate_direct()`` and records the semantic score.

        Отправляет *message* N раз в свежих сессиях, собирая Distribution.
        """
        cfg = get_config().statistical
        n = n or cfg.default_n_runs
        kw = init_kwargs or {}
        eval_threshold = get_config().evaluate.default_threshold

        if parallel and client_factory is None:
            raise ValueError(
                "parallel=True requires client_factory to avoid shared-client races"
            )
        if evaluate_metric and self._judge is None:
            raise ValueError(
                "evaluate_metric requires a configured LLM judge"
            )

        def _one_run() -> RunResult:
            client = client_factory() if client_factory is not None else self._client
            s = AgentSession(
                client=client,
                judge=self._judge,
                registry=self._registry,
            )
            s.init_session(**kw)
            start = time.perf_counter()
            s.send(message)
            latency = time.perf_counter() - start

            passed = True
            error = None
            try:
                s.expect_response_ok()
            except AssertionError as exc:
                passed = False
                error = str(exc)

            score = None
            if evaluate_metric and passed and s._judge is not None:
                result = s._evaluate_direct_result(
                    metric_name=evaluate_metric,
                    threshold=eval_threshold,
                )
                s._last_eval_result = result
                score = result.score
                if not result.passed:
                    passed = False
                    if error is None:
                        error = (
                            f"evaluate_direct('{evaluate_metric}') "
                            f"score={result.score:.3f} < threshold={eval_threshold:.3f}"
                        )

            return RunResult(
                passed=passed,
                score=score,
                latency=latency,
                response_text=s.last_text if s._last else "",
                error=error,
            )

        return run_n_times(_one_run, n, parallel=parallel)

    @staticmethod
    def expect_pass_rate(distribution: Distribution, min_rate: float) -> None:
        """Assert that pass rate >= *min_rate*. / Проверяет, что pass_rate >= min_rate."""
        assert distribution.pass_rate >= min_rate, (
            f"Pass rate {distribution.pass_rate:.2%} < required {min_rate:.2%} "
            f"({sum(1 for r in distribution.results if r.passed)}/{distribution.n} passed)"
        )

    @staticmethod
    def expect_score_ci(
        distribution: Distribution,
        min_lower_bound: float,
        confidence: float | None = None,
    ) -> None:
        """Assert that the lower bound of the CI >= *min_lower_bound*. / Проверяет, что нижняя граница CI >= min_lower_bound."""
        cfg = get_config().statistical
        conf = confidence or cfg.confidence_level
        lo, hi = distribution.confidence_interval(
            confidence=conf, n_bootstrap=cfg.bootstrap_samples,
        )
        assert lo >= min_lower_bound, (
            f"{conf:.0%} CI lower bound {lo:.3f} < required {min_lower_bound:.3f} "
            f"(CI: [{lo:.3f}, {hi:.3f}], mean={distribution.mean_score})"
        )

    # ======================================================================
    # LLM-as-Judge (DeepEval GEval)
    # ======================================================================

    def evaluate(
        self,
        metric_name: str,
        threshold: float | None = None,
        criteria: str | None = None,
        *,
        engine: str | None = None,
        context: list[str] | None = None,
        expected_output: str | None = None,
        n_samples: int | None = None,
    ) -> "AgentSession":
        """Run an LLM-as-Judge evaluation on the last response.

        *engine* selects the backend:
        - ``"geval"`` (default): DeepEval GEval — backward compatible
        - ``"direct"``: ATK G-Eval — custom pipeline per the original paper

        Запускает LLM-as-Judge оценку последнего ответа.
        """
        assert self._judge is not None, (
            "No LLM judge configured. Pass a BaseLLMJudge to AgentSession."
        )

        cfg = get_config()
        resolved_engine = engine or cfg.judge.default_engine
        logger.debug("evaluate(): resolved engine = %s", resolved_engine)

        if resolved_engine == "direct":
            return self.evaluate_direct(
                metric_name,
                threshold=threshold,
                criteria=criteria,
                context=context,
                expected_output=expected_output,
                n_samples=n_samples,
            )
        if resolved_engine != "geval":
            raise ValueError(
                f"Unsupported evaluate engine: '{resolved_engine}'. "
                "Supported values: 'geval', 'direct'"
            )
        if cfg.judge.verbose_logging or logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "evaluate(engine='geval'): detailed ATKGEval logs "
                "(raw judge responses, parsed scores, reasoning) are unavailable. "
                "Use engine='direct' or set [judge].default_engine='direct'."
            )

        try:
            from deepeval import assert_test
            from deepeval.metrics import GEval
            from deepeval.test_case import LLMTestCase, LLMTestCaseParams
        except ImportError as exc:
            raise ImportError(
                "evaluate(engine='geval') requires the deepeval package. "
                "Install with: pip install agent-test-kit[deepeval]  "
                "Or use engine='direct' for the built-in ATK G-Eval pipeline."
            ) from exc

        if criteria is None:
            criteria = self._registry.get(metric_name)
        if threshold is None:
            threshold = cfg.evaluate.default_threshold

        actual_output, last_input = self._resolve_eval_io()

        logger.debug(
            "evaluate(%s, threshold=%.2f, engine=geval)\n  input:  %s\n  output: %s",
            metric_name, threshold, last_input[:300], actual_output[:500],
        )
        if context:
            logger.debug(
                "evaluate(%s, engine=geval): context:\n%s",
                metric_name,
                self._short_repr(context, self._log_prompt_limit()),
            )
        if expected_output:
            logger.debug(
                "evaluate(%s, engine=geval): expected_output:\n%s",
                metric_name,
                self._short_text(expected_output, self._log_response_limit()),
            )

        eval_params = [LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT]
        tc_kwargs: dict[str, Any] = {"input": last_input, "actual_output": actual_output}

        if context:
            eval_params.append(LLMTestCaseParams.RETRIEVAL_CONTEXT)
            tc_kwargs["retrieval_context"] = context
        if expected_output:
            eval_params.append(LLMTestCaseParams.EXPECTED_OUTPUT)
            tc_kwargs["expected_output"] = expected_output

        metric = GEval(
            name=metric_name,
            model=self._judge,
            criteria=criteria,
            evaluation_params=eval_params,
            threshold=threshold,
        )
        test_case = LLMTestCase(**tc_kwargs)
        assert_test(test_case, [metric])
        return self

    def evaluate_direct(
        self,
        metric_name: str,
        threshold: float | None = None,
        criteria: str | None = None,
        *,
        context: list[str] | None = None,
        expected_output: str | None = None,
        n_samples: int | None = None,
    ) -> "AgentSession":
        """Evaluate using ATK G-Eval (custom pipeline per the original paper).

        Returns ``self`` for fluent chaining. The :class:`GEvalResult` is
        stored in ``last_eval_result``.

        Оценка через ATK G-Eval (собственный pipeline по оригинальной статье).
        """
        cfg = get_config()
        resolved_threshold = threshold or cfg.evaluate.default_threshold
        result = self._evaluate_direct_result(
            metric_name=metric_name,
            threshold=resolved_threshold,
            criteria=criteria,
            context=context,
            expected_output=expected_output,
            n_samples=n_samples,
        )
        self._last_eval_result = result

        assert result.passed, (
            f"Turn {self._turn}: evaluate_direct('{metric_name}') "
            f"score={result.score:.3f} < threshold={resolved_threshold:.3f}\n"
            f"Raw scores: {result.raw_scores}\n"
            f"Reasoning: {result.reasoning[:500]}"
        )
        return self

    def _build_direct_evaluator(self, n_samples: int | None = None) -> Any:
        """Build a configured ATK G-Eval evaluator instance."""
        from agent_test_kit.geval import ATKGEval

        cfg = get_config().judge
        assert self._judge is not None, (
            "No LLM judge configured. Pass a BaseLLMJudge to AgentSession."
        )
        return ATKGEval(
            judge=self._judge,
            n_samples=n_samples or cfg.n_samples,
            score_scale=cfg.score_scale,
            temperature=cfg.temperature,
            cot_cache_dir=cfg.cot_cache_dir,
            system_prompt_version=cfg.system_prompt_version,
            require_reasoning=cfg.require_reasoning,
            reasoning_backfill_attempts=cfg.reasoning_backfill_attempts,
            verbose_logging=cfg.verbose_logging,
            log_prompt_chars=cfg.log_prompt_chars,
            log_response_chars=cfg.log_response_chars,
        )

    def _evaluate_direct_result(
        self,
        metric_name: str,
        threshold: float,
        criteria: str | None = None,
        *,
        context: list[str] | None = None,
        expected_output: str | None = None,
        n_samples: int | None = None,
    ) -> Any:
        """Run ATK G-Eval and return raw result without assertions."""
        assert self._judge is not None, (
            "No LLM judge configured. Pass a BaseLLMJudge to AgentSession."
        )
        judge_cls = type(self._judge)
        judge_name = judge_cls.__name__
        try:
            model_name = self._judge.get_model_name()
        except Exception:
            model_name = "<unknown>"
        logger.debug(
            "ATKGEval judge: class=%s module=%s model=%s",
            judge_name,
            judge_cls.__module__,
            model_name,
        )
        if (
            judge_cls.__module__.startswith("tests.")
            or judge_cls.__module__.startswith("test_")
            or judge_cls.__module__ == "__main__"
        ):
            logger.warning(
                "ATKGEval is running with a mock/test judge (%s). "
                "Reasoning and scores may be synthetic.",
                judge_name,
            )
        if criteria is None:
            criteria = self._registry.get(metric_name)

        actual_output, last_input = self._resolve_eval_io()
        logger.debug(
            "_evaluate_direct_result(%s, threshold=%.2f)\n  input:  %s\n  output: %s",
            metric_name, threshold, last_input[:300], actual_output[:500],
        )
        if context:
            logger.debug(
                "_evaluate_direct_result(%s): context:\n%s",
                metric_name,
                self._short_repr(context, self._log_prompt_limit()),
            )
        if expected_output:
            logger.debug(
                "_evaluate_direct_result(%s): expected_output:\n%s",
                metric_name,
                self._short_text(expected_output, self._log_response_limit()),
            )
        evaluator = self._build_direct_evaluator(n_samples=n_samples)
        result = evaluator.evaluate(
            input_text=last_input,
            output_text=actual_output,
            criteria=criteria,
            threshold=threshold,
            metric_name=metric_name,
            context=context,
            expected_output=expected_output,
        )
        if self._should_log_debug():
            logger.debug(
                "_evaluate_direct_result(%s): final score=%.3f raw_scores=%s passed=%s",
                metric_name,
                result.score,
                result.raw_scores,
                result.passed,
            )
            logger.debug(
                "_evaluate_direct_result(%s): reasoning:\n%s",
                metric_name,
                self._short_text(str(result.reasoning), self._log_response_limit()),
            )
        return result

    def _resolve_eval_io(self) -> tuple[str, str]:
        """Extract (actual_output, last_input) for evaluation. / Извлекает (actual_output, last_input) для оценки."""
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
        return actual_output, last_input

    def _should_log_debug(self) -> bool:
        cfg = get_config().judge
        return cfg.verbose_logging or logger.isEnabledFor(logging.DEBUG)

    def _log_prompt_limit(self) -> int:
        return get_config().judge.log_prompt_chars

    def _log_response_limit(self) -> int:
        return get_config().judge.log_response_chars

    @staticmethod
    def _short_text(text: str, limit: int) -> str:
        if limit <= 0 or len(text) <= limit:
            return text
        return f"{text[:limit]}...<truncated {len(text) - limit} chars>"

    @classmethod
    def _short_repr(cls, value: Any, limit: int) -> str:
        text = repr(value)
        return cls._short_text(text, limit)

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

    @property
    def last_eval_result(self) -> Any:
        """Last :class:`GEvalResult` from ``evaluate_direct()``. / Последний GEvalResult из evaluate_direct()."""
        return self._last_eval_result

    def to_trace_dict(self) -> dict[str, Any]:
        """Return a structured session snapshot for diagnostics/reporting."""
        trace: dict[str, Any] = {
            "session_id": self._client.session_id,
            "turn": self._turn,
            "init_data": self._init_data,
            "init_message": self._init_message,
            "history": list(self._history),
            "timings": list(self._timings),
            "last_response": None,
            "last_eval_result": None,
        }
        if self._last is not None:
            trace["last_response"] = {
                "status_code": self._last.status_code,
                "text": self._last.text,
                "metadata": self._last.metadata,
                "raw": self._last.raw,
            }
        if self._last_eval_result is not None:
            result = self._last_eval_result
            trace["last_eval_result"] = {
                "score": getattr(result, "score", None),
                "raw_scores": list(getattr(result, "raw_scores", []) or []),
                "passed": getattr(result, "passed", None),
                "reasoning": getattr(result, "reasoning", ""),
                "evaluation_steps": list(getattr(result, "evaluation_steps", []) or []),
                "raw_responses": list(getattr(result, "raw_responses", []) or []),
            }
        return trace

    # ======================================================================
    # Internal
    # ======================================================================

    def _need_response(self) -> None:
        """Ensure a response exists; raise if not. / Проверяет наличие ответа; вызывает исключение при отсутствии."""
        assert self._last is not None, (
            "No response from agent. Call send() first."
        )


def run_dialogue(
    session: "AgentSession",
    messages: list[str],
    *,
    expect_ok: bool = True,
) -> "AgentSession":
    """Send every message in *messages* sequentially.

    When *expect_ok* is ``True`` (default), each turn is followed by
    ``expect_response_ok()``.

    Отправляет каждое сообщение из *messages* последовательно. При *expect_ok*
    ``True`` (по умолчанию) после каждого хода вызывается ``expect_response_ok()``.
    """
    for msg in messages:
        session.send(msg)
        if expect_ok:
            session.expect_response_ok()
    return session
