"""
Generic concurrency tests applicable to any LLM agent.

Thresholds are read from ``agent-test-kit.toml`` -> ``[concurrency]``.

Requires fixture:

- ``agent_client`` — a :class:`~agent_test_kit.BaseAgentClient` instance
  (``scope="session"``)

Универсальные тесты параллельности для любого LLM-агента.
Пороги берутся из ``agent-test-kit.toml`` -> ``[concurrency]``.
Требуется фикстура ``agent_client``.
"""
from __future__ import annotations

import threading
import time
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


class GenericConcurrencyTests:
    """Parallel-session runtime tests.

    Тесты выполнения параллельных сессий.
    """

    @staticmethod
    def _make_isolated_client(agent_client):
        clone = getattr(agent_client, "clone", None)
        if callable(clone):
            return clone()

        cls = type(agent_client)
        try:
            return cls(
                base_url=agent_client.base_url,
                timeout=agent_client.timeout,
                verify=agent_client.verify,
                log_payloads=getattr(agent_client, "_log_payloads", False),
            )
        except Exception as exc:
            pytest.skip(
                "Concurrency tests require a cloneable client or a constructor "
                f"compatible with BaseAgentClient kwargs: {exc}"
            )

    @staticmethod
    def _preview_text(text: str | None, limit: int = 120) -> str | None:
        if text is None:
            return None
        compact = " ".join(text.split())
        if len(compact) <= limit:
            return compact
        return compact[: limit - 3] + "..."

    @staticmethod
    def _short_error(message: str, limit: int = 180) -> str:
        compact = " ".join(message.split())
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
            )
        )

    def _classify_error(self, message: str) -> tuple[str, str]:
        lower = message.lower()
        if any(
            token in lower
            for token in (
                "proxyerror",
                "httpconnectionpool",
                "max retries exceeded",
                "connection reset",
                "connection refused",
            )
        ):
            return (
                "Проблема сетевого доступа",
                "Запрос не дошёл до сервиса из-за прокси, сетевого маршрута или сброса соединения.",
            )
        if any(
            token in lower
            for token in (
                "502",
                "503",
                "504",
                "bad gateway",
                "service unavailable",
                "gateway timeout",
            )
        ):
            return (
                "Сервис недоступен",
                "Сервис или его upstream вернул серверную ошибку, поэтому сценарий нельзя оценить по бизнес-логике.",
            )
        if "timed out" in lower or "timeout" in lower:
            return (
                "Превышен таймаут",
                "Одна из параллельных операций выполнялась дольше допустимого времени.",
            )
        return (
            "Ошибка выполнения worker-сессии",
            "Параллельная сессия завершилась исключением, не связанным напрямую с бизнес-результатом сценария.",
        )

    def _build_worker_report(self, run: dict[str, Any]) -> str:
        lines = [
            "# Итоги по параллельным сессиям",
            "",
            "| Worker | Статус | Время, с | Категория | Краткий ответ | Ошибка |",
            "|---|---|---:|---|---|---|",
        ]
        for worker in run["workers"]:
            status = worker.get("status", "unknown")
            elapsed = worker.get("elapsed")
            error = worker.get("error")
            category = "-"
            if error:
                category = self._classify_error(str(error))[0]
            elapsed_text = "-" if elapsed is None else f"{elapsed:.2f}"
            preview = worker.get("response_preview") or "-"
            error_text = "-"
            if error:
                error_text = self._short_error(str(error), limit=90)
            lines.append(
                "| {worker} | {status} | {elapsed} | {category} | {preview} | {error} |".format(
                    worker=worker.get("worker", "-"),
                    status=status,
                    elapsed=elapsed_text,
                    category=category,
                    preview=preview.replace("|", "/"),
                    error=error_text.replace("|", "/"),
                )
            )
        return "\n".join(lines) + "\n"

    def _build_key_findings(self, run: dict[str, Any]) -> list[str]:
        findings: list[str] = []
        if run["alive"]:
            findings.append(
                f"Не завершились потоки: {', '.join(run['alive'])}"
            )
        if run["errors"]:
            categories: dict[str, int] = {}
            explanations: dict[str, str] = {}
            for _, error in run["errors"]:
                category, explanation = self._classify_error(error)
                categories[category] = categories.get(category, 0) + 1
                explanations.setdefault(category, explanation)
            for category, count in categories.items():
                findings.append(
                    f"{category}: {count} worker(ов). {explanations[category]}"
                )
        if not run["errors"] and not run["alive"]:
            findings.append("Все worker-потоки завершились без ошибок.")
        return findings

    def _parallel_verdict(
        self,
        run: dict[str, Any],
        *,
        budget_checked: bool,
    ) -> tuple[str, str]:
        if run["alive"]:
            return (
                "Проблема с завершением параллельной обработки",
                "Не все worker-потоки завершились в пределах таймаута. "
                "Это признак зависания или слишком медленной обработки.",
            )

        if run["errors"]:
            if any(self._looks_like_environment_error(err) for _, err in run["errors"]):
                return (
                    "Проблема окружения или доступности сервиса",
                    "Проверка не дошла до оценки поведения при параллельной нагрузке, "
                    "потому что часть сессий завершилась сетевой или серверной ошибкой.",
                )
            return (
                "Ошибка выполнения параллельного сценария",
                "Часть параллельных сессий завершилась с ошибкой, поэтому сценарий "
                "нельзя считать успешно пройденным.",
            )

        missing_results = sum(result is None for result in run["results"])
        if missing_results:
            return (
                "Неполный результат",
                "Часть параллельных сессий завершилась без ответа, поэтому итог "
                "проверки нельзя считать надёжным.",
            )

        if budget_checked and run["elapsed"] >= run["wall_clock_budget"]:
            return (
                "Превышен общий лимит времени",
                "Все сессии завершились, но суммарное время обработки оказалось "
                "выше допустимого порога.",
            )

        return (
            "Проверка пройдена",
            "Параллельные сессии завершились независимо и в ожидаемых пределах "
            "по стабильности и времени выполнения.",
        )

    def _attach_parallel_report(
        self,
        *,
        title: str,
        scenario: str,
        expected: list[str],
        run: dict[str, Any],
        budget_checked: bool,
    ) -> None:
        verdict_title, verdict_text = self._parallel_verdict(
            run,
            budget_checked=budget_checked,
        )
        completed = sum(1 for worker in run["workers"] if worker["status"] == "passed")
        missing_results = sum(result is None for result in run["results"])

        actual = [
            f"Запущено параллельных сессий: {run['parallel_count']}",
            f"Успешно завершилось: {completed}",
            f"Ошибок в worker-потоках: {len(run['errors'])}",
            f"Сессий без ответа: {missing_results}",
            (
                f"Зависшие потоки: {', '.join(run['alive'])}"
                if run["alive"]
                else "Зависших потоков нет"
            ),
            f"Общее время выполнения: {run['elapsed']:.2f} с",
            f"Таймаут ожидания потока: {run['parallel_timeout']:.2f} с",
        ]
        if budget_checked:
            actual.append(
                f"Лимит общего времени: {run['wall_clock_budget']:.2f} с"
            )
        if run["errors"]:
            actual.append("Ключевые ошибки:")
            for idx, error in run["errors"][:3]:
                actual.append(f"- worker {idx}: {self._short_error(error)}")
            if len(run["errors"]) > 3:
                actual.append(f"- ...ещё ошибок: {len(run['errors']) - 3}")

        summary_lines = [
            f"Сценарий:\n{scenario}",
            "",
            "Ожидалось:",
        ]
        summary_lines.extend(f"- {item}" for item in expected)
        summary_lines.extend(["", "Фактически:"])
        summary_lines.extend(f"- {item}" for item in actual)
        summary_lines.extend(
            [
                "",
                f"Вывод:\n{verdict_title}. {verdict_text}",
            ]
        )
        summary = "\n".join(summary_lines)

        set_title(title)
        set_description(summary)
        attach_text("summary.txt", summary)
        attach_markdown("workers.md", self._build_worker_report(run))
        attach_json("[tech] parallel-batch.json", run)

    def _run_parallel_batch(self, agent_client) -> dict[str, Any]:
        cfg = get_config().concurrency
        results: list[str | None] = [None] * cfg.parallel_count
        errors: list[tuple[int, str]] = []
        workers: list[dict[str, Any]] = [
            {"worker": idx, "status": "pending"}
            for idx in range(cfg.parallel_count)
        ]

        def worker(idx: int) -> None:
            started = time.perf_counter()
            response_text: str | None = None
            try:
                session = AgentSession(client=self._make_isolated_client(agent_client))
                session.init_session()
                session.send(f"Hello from parallel session {idx}")
                session.expect_response_ok()
                response_text = session.last_text
                results[idx] = response_text
                workers[idx] = {
                    "worker": idx,
                    "status": "passed",
                    "elapsed": time.perf_counter() - started,
                    "response_preview": self._preview_text(response_text),
                    "error": None,
                }
            except Exception as exc:
                error_text = str(exc)
                errors.append((idx, error_text))
                workers[idx] = {
                    "worker": idx,
                    "status": "failed",
                    "elapsed": time.perf_counter() - started,
                    "response_preview": self._preview_text(response_text),
                    "error": error_text,
                }

        threads = [
            threading.Thread(target=worker, args=(i,), name=f"atk-parallel-{i}")
            for i in range(cfg.parallel_count)
        ]

        start = time.perf_counter()
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=cfg.parallel_timeout)
        elapsed = time.perf_counter() - start

        alive = [thread.name for thread in threads if thread.is_alive()]
        return {
            "parallel_count": cfg.parallel_count,
            "parallel_timeout": cfg.parallel_timeout,
            "wall_clock_budget": cfg.wall_clock_budget,
            "elapsed": elapsed,
            "results": results,
            "errors": errors,
            "alive": alive,
            "workers": workers,
        }

    @pytest.mark.slow
    def test_parallel_sessions_complete_without_errors(self, agent_client):
        """Parallel sessions complete independently without worker errors. / Параллельные сессии завершаются независимо и без ошибок в воркерах."""
        cfg = get_config().concurrency
        with step("Запуск параллельных сессий"):
            run = self._run_parallel_batch(agent_client)
            results = run["results"]
            errors = run["errors"]
            alive = run["alive"]

        with step("Подготовка понятной сводки для отчёта"):
            self._attach_parallel_report(
                title="Параллельные сессии завершаются без ошибок",
                scenario=(
                    "Одновременно запускаем несколько независимых сессий и отправляем "
                    "в каждую по одному сообщению, чтобы проверить стабильность "
                    "параллельной обработки."
                ),
                expected=[
                    "Все worker-потоки завершаются без исключений",
                    "Каждая сессия получает корректный ответ",
                    "После завершения не остаётся зависших потоков",
                ],
                run=run,
                budget_checked=False,
            )
            findings = self._build_key_findings(run)
            if findings:
                attach_text("findings.txt", "\n".join(f"- {item}" for item in findings))

        with step("Проверка отсутствия зависших потоков"):
            assert not alive, (
                f"Parallel sessions timed out after {cfg.parallel_timeout}s: {alive}"
            )
        with step("Проверка отсутствия ошибок в worker-потоках"):
            assert not errors, f"Parallel session errors: {errors}"
        with step("Проверка наличия ответа во всех сессиях"):
            assert all(result is not None for result in results), (
                f"Some parallel sessions returned no response: {results}"
            )

    @pytest.mark.slow
    def test_parallel_sessions_within_wall_clock_budget(self, agent_client):
        """Parallel batch finishes within the configured wall-clock budget. / Параллельный пакет завершается в пределах общего лимита wall-clock."""
        cfg = get_config().concurrency
        with step("Запуск параллельного пакета"):
            run = self._run_parallel_batch(agent_client)
            elapsed = run["elapsed"]
            errors = run["errors"]
            alive = run["alive"]

        with step("Подготовка понятной сводки для отчёта"):
            self._attach_parallel_report(
                title="Параллельный пакет укладывается в лимит времени",
                scenario=(
                    "Одновременно запускаем пакет независимых сессий и проверяем, "
                    "что весь параллельный прогон завершается в пределах заданного "
                    "общего времени."
                ),
                expected=[
                    "Все worker-потоки завершаются без ошибок",
                    "Не остаётся зависших потоков",
                    f"Общее время прогона меньше {cfg.wall_clock_budget:.2f} с",
                ],
                run=run,
                budget_checked=True,
            )
            findings = self._build_key_findings(run)
            if findings:
                attach_text("findings.txt", "\n".join(f"- {item}" for item in findings))

        with step("Проверка отсутствия зависших потоков"):
            assert not alive, (
                f"Parallel sessions timed out after {cfg.parallel_timeout}s: {alive}"
            )
        with step("Проверка отсутствия ошибок в worker-потоках"):
            assert not errors, f"Parallel session errors: {errors}"
        with step("Проверка общего лимита времени"):
            assert elapsed < cfg.wall_clock_budget, (
                f"Parallel wall-clock {elapsed:.2f}s > budget {cfg.wall_clock_budget:.2f}s"
            )
