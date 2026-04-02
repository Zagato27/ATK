"""
Helpers for optional Allure integration.

This module is intentionally dependency-tolerant: if ``allure-pytest`` is not
installed, all helper functions become no-ops.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
import importlib.metadata
import inspect
import json
import platform
import re
import sys
from pathlib import Path
from typing import Any

from agent_test_kit.config import Config

try:  # pragma: no cover - optional dependency
    import allure
    from allure_commons.types import AttachmentType
except Exception:  # pragma: no cover - optional dependency
    allure = None
    AttachmentType = None


_SENSITIVE_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "cookie",
    "password",
    "secret",
    "token",
}

_REDACTION_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z]{2,}"), "[REDACTED_EMAIL]"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[REDACTED_SSN]"),
    (re.compile(r"\b\d{4}\s?\d{6}\b"), "[REDACTED_PASSPORT]"),
    (re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\b"), "[REDACTED_CARD]"),
    (re.compile(r"\+?\d[\d\s\-]{9,14}\d"), "[REDACTED_PHONE]"),
    (re.compile(r"(?i)(bearer\s+)[a-z0-9._\-]+"), r"\1[REDACTED]"),
]


def allure_enabled() -> bool:
    return allure is not None and AttachmentType is not None


def get_allure_results_dir(config: Any) -> Path | None:
    if not allure_enabled():
        return None
    option = getattr(config, "option", None)
    raw = getattr(option, "allure_report_dir", None)
    if not raw:
        return None
    path = Path(str(raw))
    path.mkdir(parents=True, exist_ok=True)
    return path


def sanitize_for_allure(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _sanitize_text(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            key_str = str(key)
            if any(secret in key_str.lower() for secret in _SENSITIVE_KEYS):
                sanitized[key_str] = "[REDACTED]"
            else:
                sanitized[key_str] = sanitize_for_allure(item)
        return sanitized
    if isinstance(value, (list, tuple, set)):
        return [sanitize_for_allure(item) for item in value]
    return repr(value)


def attach_json(name: str, payload: Any) -> None:
    if not allure_enabled():
        return
    allure.attach(
        json.dumps(sanitize_for_allure(payload), ensure_ascii=False, indent=2),
        name=name,
        attachment_type=AttachmentType.JSON,
    )


def attach_text(name: str, text: str) -> None:
    if not allure_enabled():
        return
    allure.attach(
        _sanitize_text(text),
        name=name,
        attachment_type=AttachmentType.TEXT,
    )


def attach_markdown(name: str, text: str) -> None:
    if not allure_enabled():
        return
    attachment_type = getattr(AttachmentType, "MARKDOWN", AttachmentType.TEXT)
    allure.attach(
        _sanitize_text(text),
        name=name,
        attachment_type=attachment_type,
    )


def set_title(title: str) -> None:
    if not allure_enabled():
        return
    allure.dynamic.title(_sanitize_text(title))


def set_description(text: str) -> None:
    if not allure_enabled():
        return
    allure.dynamic.description(_sanitize_text(text))


@contextmanager
def step(title: str):
    if not allure_enabled():
        with nullcontext():
            yield
        return
    with allure.step(_sanitize_text(title)):
        yield


def set_test_labels(node: Any) -> None:
    if not allure_enabled():
        return

    suite_name = Path(str(getattr(node, "path", ""))).stem or node.nodeid.split("::", 1)[0]
    allure.dynamic.parent_suite("ATK")
    allure.dynamic.suite(suite_name)

    cls = getattr(node, "cls", None)
    if cls is not None:
        allure.dynamic.sub_suite(cls.__name__)

    feature = infer_feature_name(node)
    if feature:
        allure.dynamic.feature(feature)

    if suite_name.startswith("test_live_service_"):
        allure.dynamic.tag("live-service")
    elif suite_name == "test_generic_suites":
        allure.dynamic.tag("contract")

    for mark in node.iter_markers():
        allure.dynamic.tag(mark.name)

    obj = getattr(node, "obj", None)
    doc = inspect.getdoc(obj) if obj is not None else ""
    if doc:
        title = doc.splitlines()[0].split(" / ")[0].strip()
        if title:
            allure.dynamic.description(title)


def build_test_meta(node: Any, cfg: Config) -> dict[str, Any]:
    callspec = getattr(node, "callspec", None)
    params = sanitize_for_allure(getattr(callspec, "params", {})) if callspec else {}
    cls = getattr(node, "cls", None)
    return {
        "nodeid": node.nodeid,
        "module": getattr(getattr(node, "module", None), "__name__", ""),
        "class": cls.__name__ if cls is not None else None,
        "test_name": getattr(node, "name", ""),
        "feature": infer_feature_name(node),
        "markers": sorted({mark.name for mark in node.iter_markers()}),
        "params": params,
        "agent": {
            "base_url": cfg.agent.base_url,
            "init_path": cfg.agent.init_path,
            "chat_path": cfg.agent.chat_path,
        },
        "judge": {
            "provider": cfg.judge.provider,
            "model_name": cfg.judge.model_name,
            "api_base_url": cfg.judge.api_base_url,
        },
    }


def render_session_history(trace: dict[str, Any]) -> str:
    lines = ["# Session History", ""]
    if trace.get("session_id"):
        lines.append(f"session_id: `{trace['session_id']}`")
        lines.append("")
    if trace.get("init_message"):
        lines.append("## Init Message")
        lines.append("")
        lines.append("```text")
        lines.append(str(trace["init_message"]))
        lines.append("```")
        lines.append("")

    timings = list(trace.get("timings") or [])
    history = list(trace.get("history") or [])
    if (
        history
        and trace.get("init_message")
        and history[0].get("role") == "assistant"
        and str(history[0].get("content", "")) == str(trace["init_message"])
    ):
        history = history[1:]

    user_turn = 0
    for item in history:
        role = item.get("role", "unknown")
        content = str(item.get("content", ""))
        if role == "user":
            user_turn += 1
            lines.append(f"## Turn {user_turn} User")
        else:
            lines.append(f"## Turn {user_turn or 0} Assistant")
        lines.append("")
        lines.append("```text")
        lines.append(content)
        lines.append("```")
        if role == "assistant" and user_turn and len(timings) >= user_turn:
            lines.append("")
            lines.append(f"latency: `{timings[user_turn - 1]:.3f}s`")
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def write_environment_file(results_dir: Path, cfg: Config) -> None:
    values = {
        "ATK_VERSION": _package_version(),
        "PYTHON": sys.version.split()[0],
        "OS": platform.platform(),
        "AGENT_BASE_URL": cfg.agent.base_url,
        "AGENT_INIT_PATH": cfg.agent.init_path,
        "AGENT_CHAT_PATH": cfg.agent.chat_path,
        "JUDGE_PROVIDER": cfg.judge.provider,
        "JUDGE_MODEL": cfg.judge.model_name,
        "JUDGE_BASE_URL": cfg.judge.api_base_url or "default",
    }
    content = "\n".join(f"{key}={_sanitize_text(str(value))}" for key, value in values.items())
    (results_dir / "environment.properties").write_text(content + "\n", encoding="utf-8")


def write_categories_file(results_dir: Path) -> None:
    categories = [
        {
            "name": "Judge threshold failure",
            "matchedStatuses": ["failed"],
            "messageRegex": r".*evaluate_direct\('.*'\) score=.* < threshold=.*",
        },
        {
            "name": "HTTP / response status failure",
            "matchedStatuses": ["failed"],
            "messageRegex": r".*Turn \d+: HTTP \d+.*",
        },
        {
            "name": "Forbidden content leakage",
            "matchedStatuses": ["failed"],
            "messageRegex": r".*forbidden.*",
        },
        {
            "name": "PII leakage",
            "matchedStatuses": ["failed"],
            "messageRegex": r".*PII detected.*",
        },
        {
            "name": "Latency / budget failure",
            "matchedStatuses": ["failed"],
            "messageRegex": r".*(latency|wall-clock|timed out|byte limit).*",
        },
        {
            "name": "Length / shape failure",
            "matchedStatuses": ["failed"],
            "messageRegex": r".*(length|Jaccard|ratio).*",
        },
    ]
    (results_dir / "categories.json").write_text(
        json.dumps(categories, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def infer_feature_name(node: Any) -> str:
    cls = getattr(node, "cls", None)
    if cls is not None:
        for base in cls.mro():
            module = getattr(base, "__module__", "")
            if module.startswith("agent_test_kit.generic_tests."):
                return module.rsplit(".", 1)[-1]

    stem = Path(str(getattr(node, "path", ""))).stem or getattr(getattr(node, "module", None), "__name__", "")
    for prefix in ("test_live_service_", "test_"):
        if stem.startswith(prefix):
            stem = stem[len(prefix):]
            break
    return stem


def _package_version() -> str:
    try:
        return importlib.metadata.version("agent-test-kit")
    except importlib.metadata.PackageNotFoundError:  # pragma: no cover - editable/local
        return "unknown"


def _sanitize_text(text: str) -> str:
    result = text
    for pattern, replacement in _REDACTION_PATTERNS:
        result = pattern.sub(replacement, result)
    return result
