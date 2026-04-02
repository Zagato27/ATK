"""
AgentResponse — universal response model for LLM agent APIs.

Projects put domain-specific fields into ``metadata`` and ``raw``.

AgentResponse — универсальная модель ответа для LLM agent API.
Проекты помещают доменно-специфичные поля в ``metadata`` и ``raw``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentResponse:
    """Single response from an LLM agent.

    Attributes:
        text:        Main textual content of the response.
        status_code: HTTP status code (200 by default).
        metadata:    Structured domain-specific fields (node name, tool calls, etc.).
        raw:         Full raw JSON payload returned by the API.

    Один ответ от LLM-агента. Атрибуты: text — основной текст, status_code — HTTP-код, metadata — доменные поля, raw — полный JSON.
    """

    text: str
    status_code: int = 200
    metadata: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def tool_calls(self) -> list[dict[str, Any]]:
        """Convenience accessor for tool calls from metadata.

        Each element is a dict with at least ``name`` (str) and optionally
        ``arguments`` / ``parameters`` (dict).

        Удобный доступ к tool_calls из metadata. Каждый элемент — dict
        с ключом ``name`` и опциональными ``arguments`` / ``parameters``.
        """
        return self.metadata.get("tool_calls", [])

    @staticmethod
    def from_raw(
        data: dict[str, Any],
        *,
        text_keys: tuple[str, ...] = ("response", "message", "text", "reply", "content"),
        status_code: int = 200,
    ) -> AgentResponse:
        """Build an ``AgentResponse`` from a raw JSON dict. Tries *text_keys* in order to locate the main text field. Everything else goes into ``metadata``. / Создаёт ``AgentResponse`` из сырого JSON. Перебирает *text_keys* для поиска основного текста. Остальное — в ``metadata``."""
        text = ""
        for key in text_keys:
            if key in data and isinstance(data[key], str):
                text = data[key]
                break
        if not text:
            text = str(data)

        metadata = {k: v for k, v in data.items() if k not in text_keys}
        return AgentResponse(
            text=text,
            status_code=status_code,
            metadata=metadata,
            raw=data,
        )
