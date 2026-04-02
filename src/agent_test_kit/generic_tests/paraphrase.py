"""
Generic paraphrase-consistency tests applicable to any LLM agent.

Thresholds are read from ``agent-test-kit.toml`` -> ``[paraphrase]``.

Requires fixture:

- ``agent_client`` — a :class:`~agent_test_kit.BaseAgentClient` instance
  (``scope="session"``)

Универсальные тесты согласованности на парафразах для любого LLM-агента.
Пороги берутся из ``agent-test-kit.toml`` -> ``[paraphrase]``.
Требуется фикстура ``agent_client``.
"""
from __future__ import annotations

from typing import Any

import pytest

from agent_test_kit.allure_support import (
    attach_json,
    attach_markdown,
    attach_text,
    set_description,
    set_title,
    step,
)
from agent_test_kit.config import get_config
from agent_test_kit.session import AgentSession

_PARAPHRASES = [
    "I need help getting started",
    "Can you help me begin?",
    "I'd like to start the process please",
]


class GenericParaphraseConsistencyTests:
    """Paraphrase-consistency tests.

    Тесты согласованности ответов на парафразы одного и того же намерения.
    """

    @staticmethod
    def _preview_text(text: str | None, limit: int = 180) -> str | None:
        if text is None:
            return None
        compact = " ".join(text.split())
        if not compact:
            compact = repr(text)
        if len(compact) <= limit:
            return compact
        return compact[: limit - 3] + "..."

    @staticmethod
    def _looks_like_environment_error(message: str) -> bool:
        lower = message.lower()
        return any(
            token in lower
            for token in (
                "502",
                "503",
                "504",
                "bad gateway",
                "proxyerror",
                "httpconnectionpool",
                "max retries exceeded",
                "connection reset",
                "connection refused",
                "timed out",
                "service unavailable",
                "gateway timeout",
            )
        )

    def _classify_failure(self, exc: Exception, *, case: str) -> tuple[str, str]:
        message = str(exc)
        lower = message.lower()

        if self._looks_like_environment_error(message):
            return (
                "Проблема окружения или доступности сервиса",
                "Проверка не дошла до оценки согласованности парафразов, потому "
                "что один из запросов завершился сетевой или серверной ошибкой.",
            )
        if "response text is empty" in lower:
            return (
                "Пустой ответ",
                "На один из парафразов агент формально ответил, но не вернул "
                "пользовательского текста, поэтому сравнение ответов некорректно.",
            )
        if lower.startswith("turn") and "http " in lower:
            return (
                "Сервис вернул ошибку",
                "Сравнение парафразов нельзя интерпретировать, потому что сервис "
                "не вернул успешный HTTP-ответ.",
            )
        if case == "size" and "too far from average" in lower:
            return (
                "Размер ответов на парафразы слишком различается",
                "Один или несколько ответов оказались заметно длиннее или короче "
                "остальных, что выглядит как нестабильная реакция на одинаковое намерение.",
            )
        if case == "jaccard" and "jaccard=" in lower:
            return (
                "Ответы на парафразы слишком различаются по содержанию",
                "Лексическое пересечение между ответами оказалось ниже допустимого "
                "порога, поэтому парафразы обрабатываются недостаточно согласованно.",
            )
        return (
            "Нарушено ожидаемое поведение",
            "Результат сценария не соответствует ожидаемой согласованности ответов на парафразы.",
        )

    def _collect_paraphrase_runs(self, agent_client) -> list[dict[str, Any]]:
        runs: list[dict[str, Any]] = []
        for index, message in enumerate(_PARAPHRASES, start=1):
            with step(f"Отправка парафраза {index}"):
                session = AgentSession(client=agent_client)
                session.init_session()
                session.send(message)
                session.expect_response_ok()
                response = session.last_text
                runs.append(
                    {
                        "index": index,
                        "prompt": message,
                        "response": response,
                        "length": len(response),
                        "tokens": set(response.lower().split()),
                    }
                )
        return runs

    def _responses_markdown(self, runs: list[dict[str, Any]]) -> str:
        lines = [
            "# Ответы на парафразы",
            "",
            "| Парафраз | Длина ответа | Превью ответа |",
            "|---|---:|---|",
        ]
        for run in runs:
            lines.append(
                "| {index}. {prompt} | {length} | {preview} |".format(
                    index=run["index"],
                    prompt=str(run["prompt"]).replace("|", "/"),
                    length=run["length"],
                    preview=(self._preview_text(run["response"]) or "-").replace("|", "/"),
                )
            )
        return "\n".join(lines) + "\n"

    @staticmethod
    def _pairwise_jaccard(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        pairs: list[dict[str, Any]] = []
        for i in range(len(runs)):
            for j in range(i + 1, len(runs)):
                left = runs[i]["tokens"]
                right = runs[j]["tokens"]
                union = left | right
                score = len(left & right) / len(union) if union else 1.0
                pairs.append(
                    {
                        "left": i,
                        "right": j,
                        "jaccard": score,
                    }
                )
        return pairs

    def _size_metrics_markdown(
        self,
        runs: list[dict[str, Any]],
        *,
        avg_len: float,
        min_ratio: float,
        max_ratio: float,
    ) -> str:
        lines = [
            "# Метрики размера ответов",
            "",
            f"- Средняя длина ответа: {avg_len:.2f}",
            f"- Допустимый диапазон ratio: {min_ratio:.2f} .. {max_ratio:.2f}",
            "",
            "| Парафраз | Длина | Ratio к среднему | Статус |",
            "|---|---:|---:|---|",
        ]
        for run in runs:
            ratio = run.get("length_ratio", 1.0)
            status = "ok" if min_ratio <= ratio <= max_ratio else "вне диапазона"
            lines.append(
                f"| {run['index']} | {run['length']} | {ratio:.2f} | {status} |"
            )
        return "\n".join(lines) + "\n"

    def _jaccard_metrics_markdown(
        self,
        runs: list[dict[str, Any]],
        *,
        jaccard_pairs: list[dict[str, Any]],
        threshold: float,
    ) -> str:
        lines = [
            "# Лексическая близость ответов",
            "",
            f"- Минимально допустимый Jaccard: {threshold:.2f}",
            "",
            "| Пара ответов | Jaccard | Статус |",
            "|---|---:|---|",
        ]
        for pair in jaccard_pairs:
            status = "ok" if pair["jaccard"] >= threshold else "ниже порога"
            lines.append(
                f"| {pair['left'] + 1} vs {pair['right'] + 1} | {pair['jaccard']:.2f} | {status} |"
            )
        return "\n".join(lines) + "\n"

    def _attach_case_report(
        self,
        *,
        title: str,
        scenario: str,
        expected: list[str],
        verdict_title: str,
        verdict_text: str,
        actual: list[str],
        findings: list[str],
        runs: list[dict[str, Any]],
        technical: dict[str, Any],
        metrics_markdown: str,
    ) -> None:
        summary_lines = [
            f"Сценарий:\n{scenario}",
            "",
            "Ожидалось:",
        ]
        summary_lines.extend(f"- {item}" for item in expected)
        summary_lines.extend(["", "Фактически:"])
        summary_lines.extend(f"- {item}" for item in actual)
        summary_lines.extend(["", f"Вывод:\n{verdict_title}. {verdict_text}"])
        summary = "\n".join(summary_lines)

        set_title(title)
        set_description(summary)
        attach_text("summary.txt", summary)
        attach_markdown("responses.md", self._responses_markdown(runs))
        attach_markdown("metrics.md", metrics_markdown)
        if findings:
            attach_text("findings.txt", "\n".join(f"- {item}" for item in findings))
        attach_json("[tech] paraphrase-case.json", technical)

    @pytest.mark.slow
    def test_paraphrase_response_size_consistency(self, agent_client):
        """Paraphrased requests yield similarly-sized responses. / Парафразы запросов дают ответы сопоставимого размера."""
        cfg = get_config().paraphrase
        runs: list[dict[str, Any]] = []
        avg_len = 0.0

        try:
            with step("Сбор ответов на набор парафразов"):
                runs = self._collect_paraphrase_runs(agent_client)

            with step("Проверка сопоставимости размеров ответов"):
                lengths = [run["length"] for run in runs]
                avg_len = sum(lengths) / len(lengths)
                for i, run in enumerate(runs):
                    ratio = run["length"] / avg_len if avg_len else 1.0
                    run["length_ratio"] = ratio
                    assert cfg.length_ratio_min <= ratio <= cfg.length_ratio_max, (
                        f"Paraphrase {i}: length {run['length']} is too far from "
                        f"average {avg_len:.0f} (ratio={ratio:.2f}, "
                        f"allowed [{cfg.length_ratio_min}..{cfg.length_ratio_max}])"
                    )
        except Exception as exc:
            verdict_title, verdict_text = self._classify_failure(exc, case="size")
            worst_ratio = None
            if runs:
                ratios = [run.get("length_ratio", run["length"] / avg_len if avg_len else 1.0) for run in runs]
                worst_ratio = max(ratios, key=lambda value: abs(value - 1.0))
            self._attach_case_report(
                title="Парафразы дают ответы сопоставимого размера",
                scenario=(
                    "Один и тот же пользовательский смысл формулируется тремя "
                    "разными способами. Ответы агента должны оставаться "
                    "примерно одинаковыми по размеру."
                ),
                expected=[
                    "На каждый парафраз агент возвращает корректный ответ",
                    f"Отношение длины каждого ответа к среднему находится в диапазоне {cfg.length_ratio_min:.2f} .. {cfg.length_ratio_max:.2f}",
                ],
                verdict_title=verdict_title,
                verdict_text=verdict_text,
                actual=[
                    f"Количество успешно собранных ответов: {len(runs)}",
                    (
                        f"Средняя длина ответа: {avg_len:.2f}"
                        if runs
                        else "Средняя длина ответа: не рассчитана"
                    ),
                    (
                        f"Наиболее проблемный ratio: {worst_ratio:.2f}"
                        if worst_ratio is not None
                        else "Наиболее проблемный ratio: не рассчитан"
                    ),
                    f"Ошибка: {self._preview_text(str(exc), limit=220)}",
                ],
                findings=[
                    "Размеры ответов на парафразы отличаются сильнее допустимого диапазона или один из ответов не прошёл базовую проверку.",
                ],
                runs=runs,
                technical={
                    "case": "paraphrase_response_size_consistency",
                    "length_ratio_min": cfg.length_ratio_min,
                    "length_ratio_max": cfg.length_ratio_max,
                    "avg_len": avg_len,
                    "runs": runs,
                },
                metrics_markdown=self._size_metrics_markdown(
                    runs,
                    avg_len=avg_len,
                    min_ratio=cfg.length_ratio_min,
                    max_ratio=cfg.length_ratio_max,
                ) if runs else "# Метрики размера ответов\n\nНедостаточно данных для построения таблицы.\n",
            )
            raise

        self._attach_case_report(
            title="Парафразы дают ответы сопоставимого размера",
            scenario=(
                "Один и тот же пользовательский смысл формулируется тремя "
                "разными способами. Ответы агента должны оставаться "
                "примерно одинаковыми по размеру."
            ),
            expected=[
                "На каждый парафраз агент возвращает корректный ответ",
                f"Отношение длины каждого ответа к среднему находится в диапазоне {cfg.length_ratio_min:.2f} .. {cfg.length_ratio_max:.2f}",
            ],
            verdict_title="Проверка пройдена",
            verdict_text=(
                "Агент отвечает на разные формулировки одного намерения "
                "сопоставимыми по объёму ответами."
            ),
            actual=[
                f"Количество собранных ответов: {len(runs)}",
                f"Средняя длина ответа: {avg_len:.2f}",
            ],
            findings=[
                "Размеры ответов на парафразы остаются в ожидаемом диапазоне.",
            ],
            runs=runs,
            technical={
                "case": "paraphrase_response_size_consistency",
                "length_ratio_min": cfg.length_ratio_min,
                "length_ratio_max": cfg.length_ratio_max,
                "avg_len": avg_len,
                "runs": runs,
            },
            metrics_markdown=self._size_metrics_markdown(
                runs,
                avg_len=avg_len,
                min_ratio=cfg.length_ratio_min,
                max_ratio=cfg.length_ratio_max,
            ),
        )

    @pytest.mark.slow
    def test_paraphrase_response_jaccard_consistency(self, agent_client):
        """Paraphrased requests still yield lexically related answers. / Парафразы всё ещё дают лексически связанные ответы."""
        cfg = get_config().paraphrase
        runs: list[dict[str, Any]] = []
        jaccard_pairs: list[dict[str, Any]] = []

        try:
            with step("Сбор ответов на набор парафразов"):
                runs = self._collect_paraphrase_runs(agent_client)

            with step("Расчёт попарной лексической близости ответов"):
                jaccard_pairs = self._pairwise_jaccard(runs)
                for pair in jaccard_pairs:
                    assert pair["jaccard"] >= cfg.jaccard_min, (
                        f"Paraphrases {pair['left']} and {pair['right']}: "
                        f"Jaccard={pair['jaccard']:.2f} < {cfg.jaccard_min}"
                    )
        except Exception as exc:
            verdict_title, verdict_text = self._classify_failure(exc, case="jaccard")
            min_jaccard = (
                min(pair["jaccard"] for pair in jaccard_pairs)
                if jaccard_pairs
                else None
            )
            self._attach_case_report(
                title="Парафразы дают лексически связанные ответы",
                scenario=(
                    "Один и тот же смысл задаётся разными формулировками. "
                    "Ответы агента должны оставаться достаточно похожими "
                    "по словарному составу."
                ),
                expected=[
                    "На каждый парафраз агент возвращает корректный ответ",
                    f"Попарный Jaccard между ответами не ниже {cfg.jaccard_min:.2f}",
                ],
                verdict_title=verdict_title,
                verdict_text=verdict_text,
                actual=[
                    f"Количество успешно собранных ответов: {len(runs)}",
                    (
                        f"Минимальный Jaccard: {min_jaccard:.2f}"
                        if min_jaccard is not None
                        else "Минимальный Jaccard: не рассчитан"
                    ),
                    f"Ошибка: {self._preview_text(str(exc), limit=220)}",
                ],
                findings=[
                    "Ответы на парафразы оказались лексически более разрозненными, чем допускает порог, либо один из ответов не прошёл базовую проверку.",
                ],
                runs=runs,
                technical={
                    "case": "paraphrase_response_jaccard_consistency",
                    "jaccard_min": cfg.jaccard_min,
                    "pairs": jaccard_pairs,
                    "runs": runs,
                },
                metrics_markdown=self._jaccard_metrics_markdown(
                    runs,
                    jaccard_pairs=jaccard_pairs,
                    threshold=cfg.jaccard_min,
                ) if jaccard_pairs else "# Лексическая близость ответов\n\nНедостаточно данных для построения таблицы.\n",
            )
            raise

        self._attach_case_report(
            title="Парафразы дают лексически связанные ответы",
            scenario=(
                "Один и тот же смысл задаётся разными формулировками. "
                "Ответы агента должны оставаться достаточно похожими "
                "по словарному составу."
            ),
            expected=[
                "На каждый парафраз агент возвращает корректный ответ",
                f"Попарный Jaccard между ответами не ниже {cfg.jaccard_min:.2f}",
            ],
            verdict_title="Проверка пройдена",
            verdict_text=(
                "Ответы на разные формулировки одного намерения остаются "
                "лексически связанными и не распадаются на несвязанные варианты."
            ),
            actual=[
                f"Количество собранных ответов: {len(runs)}",
                f"Минимальный Jaccard: {min(pair['jaccard'] for pair in jaccard_pairs):.2f}",
            ],
            findings=[
                "Лексическая близость ответов на парафразы остаётся выше порога.",
            ],
            runs=runs,
            technical={
                "case": "paraphrase_response_jaccard_consistency",
                "jaccard_min": cfg.jaccard_min,
                "pairs": jaccard_pairs,
                "runs": runs,
            },
            metrics_markdown=self._jaccard_metrics_markdown(
                runs,
                jaccard_pairs=jaccard_pairs,
                threshold=cfg.jaccard_min,
            ),
        )
