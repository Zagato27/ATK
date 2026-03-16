"""
pytest plugin — auto-registered via ``pyproject.toml`` entry_points.

Registers custom markers so ``pytest --strict-markers`` doesn't complain,
and exposes the ``atk_config`` fixture.

Плагин pytest — автоматически регистрируется через entry_points в ``pyproject.toml``.
Регистрирует пользовательские маркеры и предоставляет фикстуру ``atk_config``.
"""
from __future__ import annotations

import pytest

from agent_test_kit.config import Config, get_config


def pytest_configure(config) -> None:  # type: ignore[no-untyped-def]
    """Register custom markers for pytest. / Регистрирует пользовательские маркеры для pytest."""
    config.addinivalue_line(
        "markers",
        "judge: test requires an LLM judge (expensive, slower)",
    )
    config.addinivalue_line(
        "markers",
        "slow: test takes >30 seconds (E2E, performance, stability)",
    )


@pytest.fixture(scope="session")
def atk_config() -> Config:
    """Exposes the loaded :class:`Config` as a pytest fixture. Usage in tests or other fixtures: def test_example(atk_config): assert atk_config.performance.first_message_latency > 0 / Предоставляет загруженную :class:`Config` как фикстуру pytest. Использование: def test_example(atk_config): assert atk_config.performance.first_message_latency > 0"""
    return get_config()
