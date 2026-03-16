"""
KaskoClient — HTTP client for the KASKO insurance agent API.

Demonstrates how to extend :class:`BaseAgentClient` for a specific agent.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv

from agent_test_kit import BaseAgentClient, AgentResponse

load_dotenv()

AGENT_BASE_URL = os.getenv(
    "AGENT_BASE_URL",
    "https://gigainsurance-python.apps.example.com",
)
_verify = os.getenv("AGENT_VERIFY_SSL", "false").lower()
AGENT_VERIFY_SSL: bool | str = _verify if _verify not in ("true", "false") else _verify == "true"
DEFAULT_EPK_ID = int(os.getenv("DEFAULT_EPK_ID", "111"))


# -- domain models ----------------------------------------------------------

@dataclass
class Vehicle:
    vin: str
    mark: str
    model: str
    year: int
    has_osago: bool = False
    has_kasko: bool = False

    @staticmethod
    def from_dict(d: dict) -> Vehicle:
        return Vehicle(
            vin=d.get("vin", ""),
            mark=d.get("mark", ""),
            model=d.get("model", ""),
            year=d.get("year", 0),
            has_osago=d.get("has_osago", False),
            has_kasko=d.get("has_kasko", False),
        )


@dataclass
class InitData:
    """Parsed init response for KASKO agent."""
    session_id: str
    user_name: str
    vehicles: list[Vehicle]
    message: str
    raw: dict = field(default_factory=dict)


# -- client -----------------------------------------------------------------

class KaskoClient(BaseAgentClient):
    """HTTP client for KASKO insurance agent REST API."""

    def __init__(
        self,
        base_url: str = AGENT_BASE_URL,
        timeout: int = 60,
        verify: bool | str = AGENT_VERIFY_SSL,
    ):
        super().__init__(base_url=base_url, timeout=timeout, verify=verify)
        self.last_init: InitData | None = None

    def create_session(self, **kwargs: Any) -> dict[str, Any]:
        epk_id = kwargs.get("epk_id", DEFAULT_EPK_ID)
        resp = self._post("/api/session/init", {"epk_id": epk_id})
        resp.raise_for_status()
        data = resp.json()

        if not data.get("session_id"):
            raise ValueError(f"session_id missing. Response: {data}")

        self.session_id = data["session_id"]
        self.last_init = InitData(
            session_id=data["session_id"],
            user_name=data.get("user_name", ""),
            vehicles=[Vehicle.from_dict(v) for v in data.get("vehicles", [])],
            message=data.get("message", ""),
            raw=data,
        )
        return data

    def send_message(self, message: str, **kwargs: Any) -> AgentResponse:
        if not self.session_id:
            raise RuntimeError("Session not initialized. Call create_session() first.")

        resp = self._post("/api/chat", {
            "session_id": self.session_id,
            "message": message,
        })
        resp.raise_for_status()
        return AgentResponse.from_raw(resp.json(), status_code=resp.status_code)

    def reset(self) -> None:
        super().reset()
        self.last_init = None
