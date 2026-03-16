"""
Generic performance tests applicable to any LLM agent.

All thresholds are read from ``agent-test-kit.toml`` → ``[performance]``.

Requires fixtures:

- ``session`` — initialized :class:`~agent_test_kit.AgentSession`
- ``agent_client`` — a :class:`~agent_test_kit.BaseAgentClient` instance
  (``scope="session"``)

Универсальные тесты производительности для любого LLM-агента.
Все пороги берутся из ``agent-test-kit.toml`` → ``[performance]``.
Требуются фикстуры: ``session``, ``agent_client``.
"""
from __future__ import annotations

import threading

import pytest

from agent_test_kit.config import get_config
from agent_test_kit.helpers import run_dialogue
from agent_test_kit.session import AgentSession


class GenericPerformanceTests:
    """Performance and latency tests.

    Thresholds come from ``[performance]`` in ``agent-test-kit.toml``.

    Тесты производительности и задержки. Пороги берутся из ``[performance]``.
    """

    # -- first message latency ----------------------------------------------

    @pytest.mark.slow
    def test_first_message_latency(self, session):
        """First message responds within the configured threshold. / Первое сообщение отвечает в пределах заданного порога."""
        cfg = get_config().performance
        session.send("Hello, let's get started")
        session.expect_response_ok()
        session.expect_latency_under(cfg.first_message_latency)

    # -- subsequent message latency -----------------------------------------

    @pytest.mark.slow
    def test_subsequent_message_latency(self, session):
        """Follow-up messages respond within the configured threshold. / Последующие сообщения отвечают в пределах заданного порога."""
        cfg = get_config().performance
        session.send("Hello")
        session.expect_response_ok()

        session.send("What can you do?")
        session.expect_response_ok()
        session.expect_latency_under(cfg.subsequent_message_latency)

        session.send("Tell me more")
        session.expect_response_ok()
        session.expect_latency_under(cfg.subsequent_message_latency)

    # -- session persistence ------------------------------------------------

    @pytest.mark.slow
    def test_session_alive_after_5_turns(self, session):
        """Session remains alive after 5 exchanges. / Сессия остаётся активной после 5 обменов репликами."""
        run_dialogue(session, [
            "Message 1",
            "Message 2",
            "Message 3",
            "Message 4",
            "Message 5",
        ])
        session.expect_session_alive()

    @pytest.mark.slow
    def test_session_alive_after_10_turns(self, session):
        """Session remains alive after 10 exchanges. / Сессия остаётся активной после 10 обменов репликами."""
        run_dialogue(session, [f"Message {i}" for i in range(1, 11)])
        session.expect_session_alive()

    # -- parallel sessions --------------------------------------------------

    @pytest.mark.slow
    def test_parallel_sessions(self, agent_client):
        """Multiple parallel sessions respond independently. / Несколько параллельных сессий отвечают независимо."""
        cfg = get_config().performance
        results: list[str | None] = [None] * cfg.parallel_count
        errors: list[tuple[int, str]] = []

        def make_client():
            """Prefer isolated client instances per thread when possible."""
            clone = getattr(agent_client, "clone", None)
            if callable(clone):
                return clone()
            cls = type(agent_client)
            try:
                return cls(
                    base_url=agent_client.base_url,
                    timeout=agent_client.timeout,
                    verify=agent_client.verify,
                )
            except Exception:
                # Fall back to shared client; BaseAgentClient keeps session_id
                # thread-local, so concurrent tests remain safe by default.
                return agent_client

        def worker(idx: int) -> None:
            # Each thread runs its own session / Каждый поток запускает свою сессию
            try:
                s = AgentSession(client=make_client())
                s.init_session()
                s.send("Hello from parallel session")
                s.expect_response_ok()
                results[idx] = s.last_text
            except Exception as exc:
                errors.append((idx, str(exc)))

        threads = [
            threading.Thread(target=worker, args=(i,))
            for i in range(cfg.parallel_count)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=cfg.parallel_timeout)

        assert not errors, f"Parallel session errors: {errors}"
        assert all(r is not None for r in results), (
            f"Some sessions returned None: {results}"
        )

    # -- response size bounds -----------------------------------------------

    def test_response_size_reasonable(self, session):
        """Agent response does not exceed the configured byte limit. / Ответ агента не превышает заданный лимит в байтах."""
        cfg = get_config().performance
        session.send("Tell me everything you can help me with")
        session.expect_response_ok()
        size = len(session.last_text.encode("utf-8"))
        assert size < cfg.max_response_bytes, (
            f"Response size {size} bytes exceeds "
            f"{cfg.max_response_bytes} byte limit"
        )

    # -- latency does not degrade over turns --------------------------------

    @pytest.mark.slow
    def test_latency_no_degradation(self, session):
        """Latency does not grow beyond the configured degradation factor. / Задержка не растёт сверх заданного коэффициента деградации."""
        cfg = get_config().performance
        messages = [f"Question {i+1}" for i in range(5)]
        run_dialogue(session, messages)

        if len(session.timings) >= 5:
            first = session.timings[0]
            last = session.timings[-1]
            if first > 0:
                ratio = last / first
                assert ratio < cfg.latency_degradation_factor, (
                    f"Latency degradation: turn 1 = {first:.2f}s, "
                    f"turn 5 = {last:.2f}s, ratio = {ratio:.1f}x "
                    f"(limit {cfg.latency_degradation_factor}x)"
                )
