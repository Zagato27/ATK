"""
Golden Set Regression — baseline comparison for LLM agent testing.

Stores expected (golden) responses and scores, compares current runs
against them, and alerts on drift beyond a configurable threshold.

Golden Set Regression — сравнение с базовыми (эталонными) ответами
для тестирования LLM-агентов. Хранит ожидаемые ответы и скоры,
сравнивает текущие прогоны с ними, сигнализирует о дрифте.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass
class GoldenCase:
    """A single golden test case. / Один эталонный тест-кейс."""

    id: str
    input: str
    expected_keywords: list[str] = field(default_factory=list)
    baseline_scores: dict[str, float] = field(default_factory=dict)
    baseline_text_hash: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DriftItem:
    """One metric's drift for a case. / Дрифт одной метрики для кейса."""

    case_id: str
    metric: str
    baseline_score: float
    current_score: float
    drift: float
    passed: bool


@dataclass
class GoldenReport:
    """Result of comparing current run vs golden set. / Результат сравнения текущего прогона с golden set."""

    items: list[DriftItem] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return all(item.passed for item in self.items)

    @property
    def drifted(self) -> list[DriftItem]:
        return [item for item in self.items if not item.passed]

    def summary(self) -> str:
        total = len(self.items)
        failed = len(self.drifted)
        lines = [f"Golden set: {total - failed}/{total} passed"]
        for d in self.drifted:
            lines.append(
                f"  DRIFT {d.case_id}/{d.metric}: "
                f"{d.baseline_score:.3f} -> {d.current_score:.3f} "
                f"(delta={d.drift:+.3f})"
            )
        return "\n".join(lines)


def load_golden(path: str | Path) -> list[GoldenCase]:
    """Load golden cases from a YAML file.

    Загружает golden-кейсы из YAML-файла.

    Expected format::

        - id: happy_path_greeting
          input: "Привет"
          expected_keywords: ["добро пожаловать"]
          baseline_scores:
            politeness: 0.85
          baseline_text_hash: "abc123..."
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Golden file not found: {p}")

    with open(p, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, list):
        raise ValueError(f"Golden file must contain a YAML list, got {type(raw).__name__}")

    cases: list[GoldenCase] = []
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(
                f"Golden item #{idx} must be a mapping, got {type(item).__name__}"
            )
        case_id = item.get("id")
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError(f"Golden item #{idx} must have non-empty string 'id'")
        baseline_scores = item.get("baseline_scores", {})
        if not isinstance(baseline_scores, dict):
            raise ValueError(
                f"Golden item '{case_id}' has invalid baseline_scores: "
                f"expected mapping, got {type(baseline_scores).__name__}"
            )
        cases.append(GoldenCase(
            id=case_id,
            input=item.get("input", ""),
            expected_keywords=item.get("expected_keywords", []),
            baseline_scores={k: float(v) for k, v in baseline_scores.items()},
            baseline_text_hash=item.get("baseline_text_hash", ""),
            metadata=item.get("metadata", {}),
        ))
    return cases


def save_golden(path: str | Path, cases: list[GoldenCase]) -> None:
    """Save golden cases to a YAML file. / Сохраняет golden-кейсы в YAML-файл."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    data = []
    for c in cases:
        entry: dict[str, Any] = {"id": c.id, "input": c.input}
        if c.expected_keywords:
            entry["expected_keywords"] = c.expected_keywords
        if c.baseline_scores:
            entry["baseline_scores"] = c.baseline_scores
        if c.baseline_text_hash:
            entry["baseline_text_hash"] = c.baseline_text_hash
        if c.metadata:
            entry["metadata"] = c.metadata
        data.append(entry)

    with open(p, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    logger.info("Saved %d golden cases to %s", len(cases), p)


def text_hash(text: str) -> str:
    """SHA-256 hash of text (first 16 hex chars). / SHA-256 хеш текста (первые 16 символов)."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def compare_run(
    current_results: dict[str, dict[str, float]],
    golden_cases: list[GoldenCase],
    drift_threshold: float = 0.15,
) -> GoldenReport:
    """Compare current run results against golden baseline.

    *current_results* maps ``case_id -> {metric_name: score}``.
    Drift = ``current - baseline``. Test fails if ``abs(drift) > threshold``.

    Сравнивает текущие результаты с golden baseline.
    """
    items: list[DriftItem] = []

    for case in golden_cases:
        case_scores = current_results.get(case.id, {})
        for metric, baseline in case.baseline_scores.items():
            current = case_scores.get(metric)
            if current is None:
                items.append(DriftItem(
                    case_id=case.id,
                    metric=metric,
                    baseline_score=baseline,
                    current_score=0.0,
                    drift=-baseline,
                    passed=False,
                ))
                continue

            drift = current - baseline
            passed = abs(drift) <= drift_threshold
            items.append(DriftItem(
                case_id=case.id,
                metric=metric,
                baseline_score=baseline,
                current_score=current,
                drift=drift,
                passed=passed,
            ))

    return GoldenReport(items=items)
