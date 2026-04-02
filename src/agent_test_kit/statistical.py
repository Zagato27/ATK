"""
Statistical testing utilities for N-run distributions.

Provides :class:`RunResult`, :class:`Distribution`, :func:`run_n_times`
and :func:`mann_whitney_u` for statistically rigorous LLM agent testing.

Утилиты статистического тестирования для N-кратных прогонов.
Предоставляет RunResult, Distribution, run_n_times и mann_whitney_u
для статистически строгого тестирования LLM-агентов.
"""
from __future__ import annotations

import logging
import math
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable

logger = logging.getLogger(__name__)


@dataclass
class RunResult:
    """Outcome of a single test run. / Результат одного прогона."""

    passed: bool
    score: float | None = None
    latency: float = 0.0
    response_text: str = ""
    error: str | None = None


@dataclass
class Distribution:
    """Aggregated statistics over N runs. / Статистика по N прогонам."""

    results: list[RunResult] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.results)

    @property
    def pass_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.passed) / self.n

    @property
    def scores(self) -> list[float]:
        return [r.score for r in self.results if r.score is not None]

    @property
    def mean_score(self) -> float | None:
        s = self.scores
        if not s:
            return None
        return sum(s) / len(s)

    @property
    def std_score(self) -> float | None:
        s = self.scores
        if len(s) < 2:
            return None
        mean = sum(s) / len(s)
        variance = sum((x - mean) ** 2 for x in s) / (len(s) - 1)
        return math.sqrt(variance)

    @property
    def mean_latency(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.latency for r in self.results) / self.n

    def confidence_interval(
        self, confidence: float = 0.95, n_bootstrap: int = 1000,
    ) -> tuple[float, float]:
        """Bootstrap CI for mean_score. / Bootstrap доверительный интервал для mean_score."""
        s = self.scores
        if not s:
            return (0.0, 0.0)
        if len(s) == 1:
            return (s[0], s[0])

        rng = random.Random(42)
        means = sorted(
            sum(rng.choices(s, k=len(s))) / len(s) for _ in range(n_bootstrap)
        )
        alpha = (1 - confidence) / 2
        lo = means[max(0, int(alpha * n_bootstrap))]
        hi = means[min(n_bootstrap - 1, int((1 - alpha) * n_bootstrap))]
        return (lo, hi)

    def is_stable(self, min_pass_rate: float, max_std: float) -> bool:
        """Check if distribution meets stability criteria. / Проверяет стабильность распределения."""
        if self.pass_rate < min_pass_rate:
            return False
        std = self.std_score
        if std is not None and std > max_std:
            return False
        return True


def run_n_times(
    fn: Callable[[], RunResult],
    n: int,
    *,
    parallel: bool = False,
    max_workers: int | None = None,
) -> Distribution:
    """Execute *fn* N times and collect a :class:`Distribution`.

    Выполняет *fn* N раз и собирает :class:`Distribution`.
    """
    results: list[RunResult] = []

    if parallel:
        workers = max_workers or min(n, 8)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(fn) for _ in range(n)]
            for fut in as_completed(futures):
                try:
                    results.append(fut.result())
                except Exception as exc:
                    results.append(RunResult(
                        passed=False, error=str(exc),
                    ))
    else:
        for i in range(n):
            try:
                results.append(fn())
            except Exception as exc:
                logger.warning("Run %d/%d failed: %s", i + 1, n, exc)
                results.append(RunResult(passed=False, error=str(exc)))

    return Distribution(results=results)


def mann_whitney_u(dist_a: Distribution, dist_b: Distribution) -> float:
    """Two-sided Mann-Whitney U test p-value for comparing score distributions.

    Returns approximate p-value using normal approximation.
    Raises ValueError if either distribution has fewer than 2 scores.

    Двусторонний тест Манна-Уитни для сравнения распределений скоров.
    """
    sa = dist_a.scores
    sb = dist_b.scores
    if len(sa) < 2 or len(sb) < 2:
        raise ValueError(
            f"Need >= 2 scores in each distribution, got {len(sa)} and {len(sb)}"
        )

    n1, n2 = len(sa), len(sb)
    combined = [(v, "a") for v in sa] + [(v, "b") for v in sb]
    combined.sort(key=lambda x: x[0])

    ranks: dict[int, float] = {}
    i = 0
    while i < len(combined):
        j = i
        while j < len(combined) and combined[j][0] == combined[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2
        for k in range(i, j):
            ranks[k] = avg_rank
        i = j

    r1 = sum(ranks[k] for k in range(len(combined)) if combined[k][1] == "a")
    u1 = r1 - n1 * (n1 + 1) / 2
    u2 = n1 * n2 - u1
    u = min(u1, u2)

    mu = n1 * n2 / 2
    sigma = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    if sigma == 0:
        return 1.0
    z = (u - mu) / sigma

    p = 2 * _norm_cdf(z)
    return min(p, 1.0)


def _norm_cdf(z: float) -> float:
    """Standard normal CDF approximation (Abramowitz & Stegun). / Аппроксимация CDF стандартного нормального распределения."""
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))
