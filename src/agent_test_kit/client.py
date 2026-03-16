"""
BaseAgentClient — abstract HTTP client for any LLM agent API.

Subclass and implement :meth:`create_session` / :meth:`send_message`
to adapt the framework to a specific agent.

BaseAgentClient — абстрактный HTTP-клиент для любого LLM agent API.
Наследуйте и реализуйте :meth:`create_session` / :meth:`send_message`
для адаптации фреймворка к конкретному агенту.
"""
from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from typing import Any

import requests

from agent_test_kit.response import AgentResponse

logger = logging.getLogger(__name__)


class BaseAgentClient(ABC):
    """Abstract base for agent HTTP clients. Provides a shared ``requests.Session`` and enforces the contract that every agent client must expose ``create_session``, ``send_message`` and ``reset``.

    Абстрактная база для HTTP-клиентов агентов. Предоставляет общую ``requests.Session`` и требует реализации ``create_session``, ``send_message`` и ``reset``.
    """

    def __init__(
        self,
        base_url: str,
        timeout: int = 60,
        verify: bool | str = False,
        headers: dict[str, str] | None = None,
        log_payloads: bool = False,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.verify = verify
        self._log_payloads = log_payloads
        self._http = requests.Session()
        self._http.verify = verify
        self._http.headers.update(headers or {
            "Content-Type": "application/json",
            "Accept": "application/json",
        })
        # Keep session state thread-local so one client instance can be used in
        # parallel tests without races on session_id.
        self._state = threading.local()
        self.session_id = None

    @property
    def session_id(self) -> str | None:
        """Current session id for the calling thread."""
        return getattr(self._state, "session_id", None)

    @session_id.setter
    def session_id(self, value: str | None) -> None:
        self._state.session_id = value

    @abstractmethod
    def create_session(self, **kwargs: Any) -> dict[str, Any]:
        """Start a new conversation session. Must set ``self.session_id`` and return the raw init payload. / Запускает новую сессию диалога. Должен установить ``self.session_id`` и вернуть сырой init payload."""

    @abstractmethod
    def send_message(self, message: str, **kwargs: Any) -> AgentResponse:
        """Send a user message and return the agent's response. / Отправляет сообщение пользователя и возвращает ответ агента."""

    def reset(self) -> None:
        """Discard current session state so a new one can be created. / Сбрасывает текущее состояние сессии для создания новой."""
        self.session_id = None

    # -- helpers available to subclasses / вспомогательные методы для подклассов ------------------------------------

    def _post(self, path: str, body: dict[str, Any]) -> requests.Response:
        """POST to *base_url/path* with logging and timeout. / Выполняет POST на *base_url/path* с логированием и таймаутом."""
        url = f"{self.base_url}{path}"
        if self._log_payloads:
            logger.debug(">>> POST %s  body=%s", url, self._sanitize_for_log(body))
        else:
            logger.debug(">>> POST %s", url)
        resp = self._http.post(url, json=body, timeout=self.timeout)
        if self._log_payloads:
            logger.debug(
                "<<< %s %s  body=%s",
                resp.status_code,
                resp.reason,
                self._response_preview(resp),
            )
        else:
            logger.debug(
                "<<< %s %s  bytes=%d",
                resp.status_code,
                resp.reason,
                len(resp.content),
            )
        return resp

    def _sanitize_for_log(self, value: Any) -> Any:
        """Best-effort payload redaction for debug logs."""
        if isinstance(value, dict):
            sanitized: dict[str, Any] = {}
            for key, nested in value.items():
                if self._is_sensitive_key(key):
                    sanitized[key] = "***REDACTED***"
                else:
                    sanitized[key] = self._sanitize_for_log(nested)
            return sanitized
        if isinstance(value, list):
            return [self._sanitize_for_log(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self._sanitize_for_log(item) for item in value)
        return value

    @staticmethod
    def _is_sensitive_key(key: str) -> bool:
        key_l = key.lower()
        sensitive_parts = (
            "password",
            "passwd",
            "secret",
            "token",
            "api_key",
            "authorization",
            "cookie",
            "session_id",
            "passport",
            "ssn",
            "card",
            "phone",
            "email",
        )
        return any(part in key_l for part in sensitive_parts)

    def _response_preview(self, resp: requests.Response) -> Any:
        """Return a compact, redacted response preview for debug logs."""
        content_type = resp.headers.get("Content-Type", "")
        if "application/json" in content_type.lower():
            try:
                return self._sanitize_for_log(resp.json())
            except ValueError:
                pass
        text = resp.text
        if len(text) > 500:
            return f"{text[:500]}...<truncated>"
        return text
