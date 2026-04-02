"""
pytest plugin — auto-registered via ``pyproject.toml`` entry_points.

Registers custom markers so ``pytest --strict-markers`` doesn't complain,
exposes the ``atk_config`` fixture, and optionally enriches runs with Allure
labels and attachments.

Плагин pytest — автоматически регистрируется через entry_points в ``pyproject.toml``.
Регистрирует пользовательские маркеры, предоставляет фикстуру ``atk_config``
и опционально обогащает прогоны метками и вложениями Allure.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from agent_test_kit.allure_support import (
    attach_json,
    attach_text,
    build_test_meta,
    get_allure_results_dir,
    render_session_history,
    set_test_labels,
    write_categories_file,
    write_environment_file,
)
from agent_test_kit.config import Config, get_config
from agent_test_kit.golden import GoldenCase, load_golden
from agent_test_kit.session import AgentSession, _install_session_observer, _restore_session_observer


def pytest_configure(config) -> None:  # type: ignore[no-untyped-def]
    """Register custom markers for pytest. / Регистрирует пользовательские маркеры для pytest."""
    config.addinivalue_line(
        "markers",
        "judge: test requires an LLM judge (expensive, slower)",
    )
    config.addinivalue_line(
        "markers",
        "slow: test takes >30 seconds (E2E, latency, concurrency, stability)",
    )
    config.addinivalue_line(
        "markers",
        "statistical: test runs N times for statistical analysis (heavier than slow)",
    )
    config.addinivalue_line(
        "markers",
        "golden: test compares results against a golden baseline",
    )


def pytest_sessionstart(session) -> None:  # type: ignore[no-untyped-def]
    """Write Allure environment/category files when enabled."""
    results_dir = get_allure_results_dir(session.config)
    if results_dir is None:
        return
    cfg = get_config()
    write_environment_file(results_dir, cfg)
    write_categories_file(results_dir)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):  # type: ignore[no-untyped-def]
    outcome = yield
    report = outcome.get_result()
    setattr(item, f"_atk_report_{report.when}", report)


@pytest.fixture(scope="session")
def atk_config() -> Config:
    """Exposes the loaded :class:`Config` as a pytest fixture. / Предоставляет загруженную :class:`Config` как фикстуру pytest."""
    return get_config()


@pytest.fixture(autouse=True)
def _atk_allure_capture(request, atk_config: Config):
    """Attach Allure metadata and session traces when Allure is enabled."""
    if get_allure_results_dir(request.config) is None:
        yield
        return

    observed_sessions: list[AgentSession] = []

    def remember(session: AgentSession) -> None:
        if not any(id(existing) == id(session) for existing in observed_sessions):
            observed_sessions.append(session)

    previous_observer = _install_session_observer(remember)
    set_test_labels(request.node)

    try:
        yield
    finally:
        _restore_session_observer(previous_observer)

        funcargs = getattr(request.node, "funcargs", {})
        fixture_session_names: dict[int, str] = {}
        for name, value in funcargs.items():
            if isinstance(value, AgentSession):
                fixture_session_names[id(value)] = name
                remember(value)

        attach_json("test-meta.json", build_test_meta(request.node, atk_config))

        for index, session in enumerate(observed_sessions, start=1):
            label = fixture_session_names.get(id(session), f"session_{index}")
            trace = session.to_trace_dict()
            attach_json(f"{label}-trace.json", trace)
            attach_text(f"{label}-history.txt", render_session_history(trace))
            if trace.get("last_eval_result") is not None:
                attach_json(f"{label}-g-eval.json", trace["last_eval_result"])

        failed_reports = [
            _serialize_report(report)
            for phase in ("setup", "call", "teardown")
            for report in [getattr(request.node, f"_atk_report_{phase}", None)]
            if report is not None and report.failed
        ]
        if failed_reports:
            attach_json("pytest-failure-report.json", failed_reports)
            for report in failed_reports:
                longrepr = report.get("longreprtext", "")
                if longrepr:
                    attach_text(f"{report['when']}-failure.txt", longrepr)


@pytest.fixture(scope="session")
def golden_set(atk_config: Config) -> list[GoldenCase]:
    """Load golden cases from the configured ``golden_dir``.

    Returns an empty list if the directory or files don't exist yet
    (allows bootstrapping).

    Загружает golden-кейсы из ``golden_dir``. Возвращает пустой список,
    если директория или файлы ещё не созданы.
    """
    golden_dir = Path(atk_config.golden.golden_dir)
    if not golden_dir.exists():
        return []

    cases: list[GoldenCase] = []
    for yaml_file in sorted(golden_dir.glob("*.yaml")):
        cases.extend(load_golden(yaml_file))
    for yml_file in sorted(golden_dir.glob("*.yml")):
        cases.extend(load_golden(yml_file))
    return cases


def _serialize_report(report: Any) -> dict[str, Any]:
    return {
        "when": report.when,
        "outcome": report.outcome,
        "duration": report.duration,
        "longreprtext": getattr(report, "longreprtext", ""),
    }
