"""
Generic latency tests applicable to any LLM agent.

Thresholds are read from ``agent-test-kit.toml`` -> ``[latency]``.

Requires fixtures:

- ``session`` — initialized :class:`~agent_test_kit.AgentSession`
- ``agent_client`` — a :class:`~agent_test_kit.BaseAgentClient` instance
  (``scope="session"``)

Универсальные тесты задержки для любого LLM-агента.
Пороги берутся из ``agent-test-kit.toml`` -> ``[latency]``.
Требуются фикстуры: ``session``, ``agent_client``.
"""
from __future__ import annotations

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
from agent_test_kit.session import AgentSession, run_dialogue


class GenericLatencyTests:
    """Latency-focused runtime tests.

    Тесты, сфокусированные на задержке ответов.
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

    def _safe_last_text(self, session) -> str | None:
        try:
            return session.last_text
        except AssertionError:
            return None

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
                "Проверка не дошла до оценки задержки, потому что запрос "
                "завершился сетевой или серверной ошибкой.",
            )
        if "response text is empty" in lower:
            return (
                "Пустой ответ",
                "Агент формально ответил, но не вернул пользовательского текста, "
                "поэтому задержку по содержательному ответу нельзя считать успешной.",
            )
        if lower.startswith("turn") and "http " in lower:
            return (
                "Сервис вернул ошибку",
                "Измерение задержки невозможно интерпретировать, потому что сервис "
                "не вернул успешный HTTP-ответ.",
            )
        if case == "init" and "init session latency" in lower:
            return (
                "Инициализация сессии медленнее порога",
                "Свежая сессия создаётся дольше, чем допускает текущий SLA.",
            )
        if case in {"first", "subsequent"} and "latency" in lower and "threshold" in lower:
            return (
                "Ответ медленнее допустимого порога",
                "Фактическая задержка ответа оказалась выше настроенного порога.",
            )
        if case == "degradation" and "latency degradation" in lower:
            return (
                "Задержка деградирует по ходу диалога",
                "Ближе к концу сценария ответы становятся заметно медленнее, чем в начале.",
            )
        if case == "init" and "session initialization did not produce session_id" in lower:
            return (
                "Инициализация не вернула session_id",
                "Сессия формально стартовала, но не выдала идентификатор, поэтому результат нельзя считать корректным.",
            )
        if "no timings recorded" in lower:
            return (
                "Нет данных по задержке",
                "После отправки сообщения не были записаны замеры времени ответа.",
            )
        return (
            "Нарушено ожидаемое поведение",
            "Результат сценария не соответствует ожидаемому поведению по задержке.",
        )

    def _build_timing_rows(
        self,
        *,
        labels: list[str],
        timings: list[float],
        thresholds: list[float | None],
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for idx, label in enumerate(labels):
            elapsed = timings[idx] if idx < len(timings) else None
            threshold = thresholds[idx] if idx < len(thresholds) else None
            if elapsed is None:
                status = "нет данных"
            elif threshold is None:
                status = "инфо"
            elif elapsed < threshold:
                status = "ok"
            else:
                status = "выше порога"
            rows.append(
                {
                    "label": label,
                    "elapsed": elapsed,
                    "threshold": threshold,
                    "status": status,
                }
            )
        return rows

    def _timings_markdown(
        self,
        *,
        rows: list[dict[str, Any]],
        degradation_ratio: float | None = None,
        degradation_limit: float | None = None,
    ) -> str:
        lines = [
            "# Замеры задержки",
            "",
            "| Этап | Время, с | Порог, с | Статус |",
            "|---|---:|---:|---|",
        ]
        for row in rows:
            elapsed = "-" if row["elapsed"] is None else f"{row['elapsed']:.2f}"
            threshold = "-" if row["threshold"] is None else f"{row['threshold']:.2f}"
            lines.append(
                f"| {row['label']} | {elapsed} | {threshold} | {row['status']} |"
            )
        if degradation_limit is not None:
            lines.extend(
                [
                    "",
                    "## Деградация",
                    "",
                    f"- Допустимый коэффициент деградации: {degradation_limit:.2f}x",
                    (
                        f"- Фактический коэффициент деградации: {degradation_ratio:.2f}x"
                        if degradation_ratio is not None
                        else "- Фактический коэффициент деградации: не рассчитан"
                    ),
                ]
            )
        return "\n".join(lines) + "\n"

    def _build_actual(
        self,
        *,
        session: AgentSession | None = None,
        extra: list[str] | None = None,
    ) -> list[str]:
        actual: list[str] = []
        if session is not None:
            actual.append(f"Количество замеров задержки: {len(session.timings)}")
            last_text = self._safe_last_text(session)
            if last_text is not None:
                actual.append(
                    f"Превью последнего ответа: {self._preview_text(last_text) or repr(last_text)}"
                )
        if extra:
            actual.extend(extra)
        return actual

    def _attach_case_report(
        self,
        *,
        title: str,
        scenario: str,
        expected: list[str],
        session: AgentSession | None,
        verdict_title: str,
        verdict_text: str,
        actual_extra: list[str] | None = None,
        findings: list[str] | None = None,
        timing_rows: list[dict[str, Any]] | None = None,
        technical: dict[str, Any] | None = None,
        degradation_ratio: float | None = None,
        degradation_limit: float | None = None,
    ) -> None:
        actual = self._build_actual(session=session, extra=actual_extra)
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
        if timing_rows:
            attach_markdown(
                "timings.md",
                self._timings_markdown(
                    rows=timing_rows,
                    degradation_ratio=degradation_ratio,
                    degradation_limit=degradation_limit,
                ),
            )
        if findings:
            attach_text("findings.txt", "\n".join(f"- {item}" for item in findings))
        if technical is not None:
            attach_json("[tech] latency-case.json", technical)

    @pytest.mark.slow
    def test_init_session_latency(self, agent_client):
        """Fresh session initialization stays within the configured threshold. / Инициализация свежей сессии укладывается в заданный порог."""
        cfg = get_config().latency
        agent_client.reset()
        session = AgentSession(client=agent_client)
        elapsed = 0.0
        timing_rows: list[dict[str, Any]] = []

        try:
            with step("Сброс клиента и подготовка новой сессии"):
                agent_client.reset()

            with step("Инициализация свежей сессии"):
                start = time.perf_counter()
                try:
                    session.init_session()
                finally:
                    elapsed = time.perf_counter() - start
                    timing_rows = self._build_timing_rows(
                        labels=["Инициализация сессии"],
                        timings=[elapsed],
                        thresholds=[cfg.init_session_latency],
                    )

            with step("Проверка порога инициализации"):
                assert elapsed < cfg.init_session_latency, (
                    f"Init session latency {elapsed:.2f}s > "
                    f"threshold {cfg.init_session_latency}s"
                )

            with step("Проверка наличия session_id"):
                assert agent_client.session_id, (
                    "Session initialization did not produce session_id"
                )
        except Exception as exc:
            verdict_title, verdict_text = self._classify_failure(exc, case="init")
            self._attach_case_report(
                title="Инициализация сессии укладывается в порог",
                scenario=(
                    "Создаём новую сессию с нуля и измеряем время её "
                    "инициализации."
                ),
                expected=[
                    f"Инициализация новой сессии быстрее {cfg.init_session_latency:.2f} с",
                    "После инициализации появляется session_id",
                ],
                session=session,
                verdict_title=verdict_title,
                verdict_text=verdict_text,
                actual_extra=[
                    f"Фактическая длительность инициализации: {elapsed:.2f} с",
                    f"Порог инициализации: {cfg.init_session_latency:.2f} с",
                    f"session_id получен: {'да' if bool(agent_client.session_id) else 'нет'}",
                    f"Ошибка: {self._preview_text(str(exc), limit=220)}",
                ],
                findings=[
                    "Свежая инициализация либо заняла слишком много времени, либо завершилась некорректно.",
                ],
                timing_rows=timing_rows,
                technical={
                    "case": "init_session_latency",
                    "elapsed": elapsed,
                    "threshold": cfg.init_session_latency,
                    "session_id": agent_client.session_id,
                },
            )
            raise

        self._attach_case_report(
            title="Инициализация сессии укладывается в порог",
            scenario=(
                "Создаём новую сессию с нуля и измеряем время её "
                "инициализации."
            ),
            expected=[
                f"Инициализация новой сессии быстрее {cfg.init_session_latency:.2f} с",
                "После инициализации появляется session_id",
            ],
            session=session,
            verdict_title="Проверка пройдена",
            verdict_text=(
                "Свежая сессия создаётся в ожидаемое время и корректно "
                "возвращает session_id."
            ),
            actual_extra=[
                f"Фактическая длительность инициализации: {elapsed:.2f} с",
                f"Порог инициализации: {cfg.init_session_latency:.2f} с",
                f"session_id получен: {'да' if bool(agent_client.session_id) else 'нет'}",
            ],
            findings=[
                "Инициализация новой сессии соответствует ожидаемому SLA.",
            ],
            timing_rows=timing_rows,
            technical={
                "case": "init_session_latency",
                "elapsed": elapsed,
                "threshold": cfg.init_session_latency,
                "session_id": agent_client.session_id,
            },
        )

    @pytest.mark.slow
    def test_first_message_latency(self, session):
        """First message responds within the configured threshold. / Первое сообщение отвечает в пределах заданного порога."""
        cfg = get_config().latency
        prompt = "Hello, let's get started"
        timing_rows: list[dict[str, Any]] = []

        try:
            with step("Отправка первого пользовательского сообщения"):
                session.send(prompt)
                timing_rows = self._build_timing_rows(
                    labels=["Первый ответ"],
                    timings=list(session.timings[-1:]),
                    thresholds=[cfg.first_message_latency],
                )

            with step("Проверка базовой корректности первого ответа"):
                session.expect_response_ok()

            with step("Проверка порога первого ответа"):
                session.expect_latency_under(cfg.first_message_latency)
        except Exception as exc:
            verdict_title, verdict_text = self._classify_failure(exc, case="first")
            self._attach_case_report(
                title="Первый ответ укладывается в порог задержки",
                scenario=(
                    "После старта сессии отправляем первое сообщение и "
                    "измеряем время первого ответа агента."
                ),
                expected=[
                    "Агент возвращает непустой первый ответ",
                    f"Первый ответ быстрее {cfg.first_message_latency:.2f} с",
                ],
                session=session,
                verdict_title=verdict_title,
                verdict_text=verdict_text,
                actual_extra=[
                    f"Порог первого ответа: {cfg.first_message_latency:.2f} с",
                    f"Ошибка: {self._preview_text(str(exc), limit=220)}",
                ],
                findings=[
                    "Первый ответ либо оказался слишком медленным, либо не прошёл базовую проверку.",
                ],
                timing_rows=timing_rows,
                technical={
                    "case": "first_message_latency",
                    "prompt": prompt,
                    "timings": list(session.timings),
                    "threshold": cfg.first_message_latency,
                    "turn": session.turn,
                },
            )
            raise

        self._attach_case_report(
            title="Первый ответ укладывается в порог задержки",
            scenario=(
                "После старта сессии отправляем первое сообщение и "
                "измеряем время первого ответа агента."
            ),
            expected=[
                "Агент возвращает непустой первый ответ",
                f"Первый ответ быстрее {cfg.first_message_latency:.2f} с",
            ],
            session=session,
            verdict_title="Проверка пройдена",
            verdict_text=(
                "Первый ответ агента приходит в допустимое время и выглядит "
                "стабильным для пользовательского сценария."
            ),
            actual_extra=[
                f"Порог первого ответа: {cfg.first_message_latency:.2f} с",
            ],
            findings=[
                "Первый ответ соответствует ожидаемому времени отклика.",
            ],
            timing_rows=timing_rows,
            technical={
                "case": "first_message_latency",
                "prompt": prompt,
                "timings": list(session.timings),
                "threshold": cfg.first_message_latency,
                "turn": session.turn,
            },
        )

    @pytest.mark.slow
    def test_subsequent_message_latency(self, session):
        """Warm follow-up messages respond within the configured threshold. / Последующие сообщения в прогретой сессии отвечают в пределах заданного порога."""
        cfg = get_config().latency
        prompts = ["Hello", "What can you do?", "Tell me more"]
        timing_rows: list[dict[str, Any]] = []

        try:
            with step("Отправка прогревающего сообщения"):
                session.send(prompts[0])
                timing_rows = self._build_timing_rows(
                    labels=[
                        "Прогрев",
                        "Последующий ответ 1",
                        "Последующий ответ 2",
                    ],
                    timings=list(session.timings),
                    thresholds=[
                        None,
                        cfg.subsequent_message_latency,
                        cfg.subsequent_message_latency,
                    ],
                )
            with step("Проверка базовой корректности прогревающего ответа"):
                session.expect_response_ok()

            with step("Отправка первого последующего сообщения"):
                session.send(prompts[1])
                timing_rows = self._build_timing_rows(
                    labels=[
                        "Прогрев",
                        "Последующий ответ 1",
                        "Последующий ответ 2",
                    ],
                    timings=list(session.timings),
                    thresholds=[
                        None,
                        cfg.subsequent_message_latency,
                        cfg.subsequent_message_latency,
                    ],
                )
            with step("Проверка первого последующего ответа"):
                session.expect_response_ok()
                session.expect_latency_under(cfg.subsequent_message_latency)

            with step("Отправка второго последующего сообщения"):
                session.send(prompts[2])
                timing_rows = self._build_timing_rows(
                    labels=[
                        "Прогрев",
                        "Последующий ответ 1",
                        "Последующий ответ 2",
                    ],
                    timings=list(session.timings),
                    thresholds=[
                        None,
                        cfg.subsequent_message_latency,
                        cfg.subsequent_message_latency,
                    ],
                )
            with step("Проверка второго последующего ответа"):
                session.expect_response_ok()
                session.expect_latency_under(cfg.subsequent_message_latency)
        except Exception as exc:
            verdict_title, verdict_text = self._classify_failure(exc, case="subsequent")
            self._attach_case_report(
                title="Последующие ответы укладываются в порог задержки",
                scenario=(
                    "После прогревающей реплики отправляем ещё два сообщения и "
                    "проверяем, что последующие ответы остаются быстрыми."
                ),
                expected=[
                    "Прогревающий ответ успешно приходит",
                    f"Каждый последующий ответ быстрее {cfg.subsequent_message_latency:.2f} с",
                ],
                session=session,
                verdict_title=verdict_title,
                verdict_text=verdict_text,
                actual_extra=[
                    f"Порог для последующих ответов: {cfg.subsequent_message_latency:.2f} с",
                    f"Количество отправленных сообщений: {len(prompts)}",
                    f"Ошибка: {self._preview_text(str(exc), limit=220)}",
                ],
                findings=[
                    "Один из последующих ответов оказался слишком медленным или не прошёл базовую проверку.",
                ],
                timing_rows=timing_rows,
                technical={
                    "case": "subsequent_message_latency",
                    "prompts": prompts,
                    "timings": list(session.timings),
                    "threshold": cfg.subsequent_message_latency,
                    "turn": session.turn,
                },
            )
            raise

        self._attach_case_report(
            title="Последующие ответы укладываются в порог задержки",
            scenario=(
                "После прогревающей реплики отправляем ещё два сообщения и "
                "проверяем, что последующие ответы остаются быстрыми."
            ),
            expected=[
                "Прогревающий ответ успешно приходит",
                f"Каждый последующий ответ быстрее {cfg.subsequent_message_latency:.2f} с",
            ],
            session=session,
            verdict_title="Проверка пройдена",
            verdict_text=(
                "После прогрева сессии агент удерживает приемлемую скорость "
                "на последующих ответах."
            ),
            actual_extra=[
                f"Порог для последующих ответов: {cfg.subsequent_message_latency:.2f} с",
                f"Количество отправленных сообщений: {len(prompts)}",
            ],
            findings=[
                "Последующие ответы укладываются в ожидаемый latency-порог.",
            ],
            timing_rows=timing_rows,
            technical={
                "case": "subsequent_message_latency",
                "prompts": prompts,
                "timings": list(session.timings),
                "threshold": cfg.subsequent_message_latency,
                "turn": session.turn,
            },
        )

    @pytest.mark.slow
    def test_latency_no_degradation(self, session):
        """Latency does not grow beyond the configured degradation factor. / Задержка не растёт сверх заданного коэффициента деградации."""
        cfg = get_config().latency
        messages = [f"Question {i + 1}" for i in range(5)]
        ratio: float | None = None
        timing_rows: list[dict[str, Any]] = []

        try:
            with step("Запуск серии из пяти последовательных сообщений"):
                run_dialogue(session, messages)
                timing_rows = self._build_timing_rows(
                    labels=[f"Ответ {i + 1}" for i in range(5)],
                    timings=list(session.timings),
                    thresholds=[None] * 5,
                )

            with step("Оценка деградации задержки"):
                if len(session.timings) >= 5:
                    first = session.timings[0]
                    last = session.timings[-1]
                    if first > 0:
                        ratio = last / first
                        assert ratio < cfg.latency_degradation_factor, (
                            f"Latency degradation: turn 1 = {first:.2f}s, "
                            f"turn 5 = {last:.2f}s, ratio = {ratio:.1f}x "
                            f"(limit {cfg.latency_degradation_factor}x)"
                        )
        except Exception as exc:
            verdict_title, verdict_text = self._classify_failure(exc, case="degradation")
            self._attach_case_report(
                title="Задержка не деградирует по мере диалога",
                scenario=(
                    "Проводим серию из пяти сообщений подряд и проверяем, "
                    "что ответы ближе к концу диалога не становятся "
                    "существенно медленнее, чем в начале."
                ),
                expected=[
                    "Серия из пяти ответов завершается без ошибок",
                    f"Коэффициент деградации ниже {cfg.latency_degradation_factor:.2f}x",
                ],
                session=session,
                verdict_title=verdict_title,
                verdict_text=verdict_text,
                actual_extra=[
                    f"Допустимый коэффициент деградации: {cfg.latency_degradation_factor:.2f}x",
                    (
                        f"Фактический коэффициент деградации: {ratio:.2f}x"
                        if ratio is not None
                        else "Фактический коэффициент деградации: не рассчитан"
                    ),
                    f"Ошибка: {self._preview_text(str(exc), limit=220)}",
                ],
                findings=[
                    "В ходе серии сообщений задержка выросла сильнее допустимого или сценарий завершился исключением.",
                ],
                timing_rows=timing_rows,
                technical={
                    "case": "latency_no_degradation",
                    "messages": messages,
                    "timings": list(session.timings),
                    "ratio": ratio,
                    "limit": cfg.latency_degradation_factor,
                    "turn": session.turn,
                },
                degradation_ratio=ratio,
                degradation_limit=cfg.latency_degradation_factor,
            )
            raise

        self._attach_case_report(
            title="Задержка не деградирует по мере диалога",
            scenario=(
                "Проводим серию из пяти сообщений подряд и проверяем, "
                "что ответы ближе к концу диалога не становятся "
                "существенно медленнее, чем в начале."
            ),
            expected=[
                "Серия из пяти ответов завершается без ошибок",
                f"Коэффициент деградации ниже {cfg.latency_degradation_factor:.2f}x",
            ],
            session=session,
            verdict_title="Проверка пройдена",
            verdict_text=(
                "По мере развития диалога скорость ответов остаётся в "
                "ожидаемых пределах и не показывает заметной деградации."
            ),
            actual_extra=[
                f"Допустимый коэффициент деградации: {cfg.latency_degradation_factor:.2f}x",
                (
                    f"Фактический коэффициент деградации: {ratio:.2f}x"
                    if ratio is not None
                    else "Фактический коэффициент деградации: не рассчитывался"
                ),
            ],
            findings=[
                "Скорость ответов остаётся стабильной по мере диалога.",
            ],
            timing_rows=timing_rows,
            technical={
                "case": "latency_no_degradation",
                "messages": messages,
                "timings": list(session.timings),
                "ratio": ratio,
                "limit": cfg.latency_degradation_factor,
                "turn": session.turn,
            },
            degradation_ratio=ratio,
            degradation_limit=cfg.latency_degradation_factor,
        )
