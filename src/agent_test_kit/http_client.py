"""
Config-driven HTTP client for a tested agent service.

Reads endpoint paths and payload defaults from ``agent-test-kit.toml``.
"""
from __future__ import annotations

from typing import Any

from agent_test_kit.client import BaseAgentClient
from agent_test_kit.config import AgentConfig, get_config
from agent_test_kit.response import AgentResponse


class ConfiguredAgentClient(BaseAgentClient):
    """HTTP client configured via ``[agent]`` section in ATK config."""

    def __init__(
        self,
        config: AgentConfig | None = None,
        *,
        base_url: str | None = None,
        timeout: int | None = None,
        verify: bool | str | None = None,
        log_payloads: bool | None = None,
    ):
        cfg = config or get_config().agent
        resolved_base_url = (base_url or cfg.base_url).rstrip("/")
        if not resolved_base_url:
            raise ValueError(
                "agent.base_url is not configured. "
                "Set it in agent-test-kit.toml under [agent]."
            )

        super().__init__(
            base_url=resolved_base_url,
            timeout=timeout or cfg.timeout,
            verify=cfg.verify_ssl if verify is None else verify,
            log_payloads=cfg.log_payloads if log_payloads is None else log_payloads,
        )
        self._config = cfg

    def create_session(self, **kwargs: Any) -> dict[str, Any]:
        payload = dict(self._config.init_payload)
        payload.update(kwargs)

        resp = self._post(self._config.init_path, payload)
        resp.raise_for_status()
        data = resp.json()

        session_id = data.get(self._config.session_id_field)
        if not session_id:
            raise ValueError(
                f"{self._config.session_id_field!r} missing in init response: {data}"
            )
        self.session_id = str(session_id)
        return data

    def send_message(self, message: str, **kwargs: Any) -> AgentResponse:
        if not self.session_id:
            raise RuntimeError("Session not initialized. Call create_session() first.")

        payload = dict(self._config.chat_static_payload)
        payload[self._config.session_id_field] = self.session_id
        payload[self._config.message_field] = message
        payload.update(kwargs)

        resp = self._post(self._config.chat_path, payload)
        resp.raise_for_status()
        return AgentResponse.from_raw(resp.json(), status_code=resp.status_code)

    def clone(self) -> "ConfiguredAgentClient":
        """Create an independent copy for latency/concurrency tests."""
        return ConfiguredAgentClient(
            config=self._config,
            base_url=self.base_url,
            timeout=self.timeout,
            verify=self.verify,
            log_payloads=self._log_payloads,
        )
