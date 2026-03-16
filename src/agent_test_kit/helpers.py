"""
Shared test helpers for agent testing.

These are designed to be imported directly into ``conftest.py`` or test files::

    from agent_test_kit.helpers import run_dialogue

Общие вспомогательные функции для тестирования агентов. Предназначены для
импорта в ``conftest.py`` или тестовые файлы.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_test_kit.session import AgentSession


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
