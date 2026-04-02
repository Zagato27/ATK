"""
Generic reproducibility tests applicable to any LLM agent.

Thresholds are read from ``agent-test-kit.toml`` -> ``[reproducibility]``.

Requires fixtures:

- ``session`` — initialized :class:`~agent_test_kit.AgentSession`
- ``agent_client`` — a :class:`~agent_test_kit.BaseAgentClient` instance
  (``scope="session"``)

Универсальные тесты воспроизводимости для любого LLM-агента.
Пороги берутся из ``agent-test-kit.toml`` -> ``[reproducibility]``.
Требуются фикстуры: ``session``, ``agent_client``.
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


class GenericReproducibilityTests:
    """Repeated-input reproducibility tests.

    Тесты воспроизводимости на повторяющихся входах.
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

    def _detected_forbidden(
        self,
        session,
        forbidden: list[str] | tuple[str, ...],
    ) -> list[str]:
        last_text = self._safe_last_text(session)
        if not last_text:
            return []
        lower = last_text.lower()
        return [item for item in forbidden if item.lower() in lower]

    def _classify_failure(
        self,
        exc: Exception,
        *,
        case: str,
        detected_forbidden: list[str] | None = None,
    ) -> tuple[str, str]:
        message = str(exc)
        lower = message.lower()
        detected_forbidden = detected_forbidden or []

        if "timed out" in lower or "timeout" in lower:
            return (
                "Превышен таймаут",
                "Проверка воспроизводимости не завершилась вовремя, поэтому "
                "результат нельзя интерпретировать надёжно.",
            )
        if self._looks_like_environment_error(message):
            return (
                "Проблема окружения или доступности сервиса",
                "Проверка не завершилась из-за сетевой или серверной ошибки, а не "
                "из-за самой воспроизводимости ответов.",
            )
        if "response text is empty" in lower:
            return (
                "Пустой ответ",
                "Один из ответов формально был получен, но без пользовательского "
                "текста, поэтому сравнение нестабильно.",
            )
        if lower.startswith("turn") and "http " in lower:
            return (
                "Сервис вернул ошибку",
                "Во время сценария воспроизводимости сервис не отдал успешный HTTP-ответ.",
            )
        if detected_forbidden:
            return (
                "В ответе появились технические артефакты",
                "Повторный ответ содержит признаки внутренней ошибки или служебные "
                "артефакты вместо стабильного пользовательского ответа.",
            )
        if case == "jaccard" and "jaccard=" in lower:
            return (
                "Независимые запуски дают слишком разные ответы",
                "Лексическое пересечение между ответами на один и тот же запрос "
                "оказалось ниже допустимого порога.",
            )
        if case == "duplicate" and "degraded unexpectedly" in lower:
            return (
                "Повторный ответ заметно деградировал",
                "После повторного ввода того же сообщения ответ стал слишком "
                "коротким или слишком длинным относительно первого ответа.",
            )
        return (
            "Нарушено ожидаемое поведение",
            "Результат проверки не соответствует ожидаемой стабильности ответов.",
        )

    def _collect_runs(
        self,
        agent_client,
        *,
        prompt: str,
        runs_count: int,
    ) -> list[dict[str, Any]]:
        runs: list[dict[str, Any]] = []
        for index in range(1, runs_count + 1):
            with step(f"Независимый запуск {index}"):
                session = AgentSession(client=agent_client)
                session.init_session()
                session.send(prompt)
                session.expect_response_ok()
                response = session.last_text
                runs.append(
                    {
                        "run": index,
                        "response": response,
                        "length": len(response),
                        "tokens": set(response.lower().split()),
                    }
                )
        return runs

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

    @staticmethod
    def _serializable_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "run": run["run"],
                "response": run["response"],
                "length": run["length"],
            }
            for run in runs
        ]

    def _runs_markdown(self, runs: list[dict[str, Any]]) -> str:
        lines = [
            "# Независимые запуски",
            "",
            "| Запуск | Длина ответа | Превью ответа |",
            "|---|---:|---|",
        ]
        for run in runs:
            lines.append(
                "| {run}. | {length} | {preview} |".format(
                    run=run["run"],
                    length=run["length"],
                    preview=(self._preview_text(run["response"]) or "-").replace("|", "/"),
                )
            )
        return "\n".join(lines) + "\n"

    def _jaccard_metrics_markdown(
        self,
        *,
        pairs: list[dict[str, Any]],
        threshold: float,
    ) -> str:
        lines = [
            "# Попарная лексическая близость",
            "",
            f"- Минимально допустимый Jaccard: {threshold:.2f}",
            "",
            "| Пара запусков | Jaccard | Статус |",
            "|---|---:|---|",
        ]
        for pair in pairs:
            status = "ok" if pair["jaccard"] >= threshold else "ниже порога"
            lines.append(
                f"| {pair['left'] + 1} vs {pair['right'] + 1} | {pair['jaccard']:.2f} | {status} |"
            )
        return "\n".join(lines) + "\n"

    def _dialogue_markdown(self, session) -> str:
        history = list(getattr(session, "history", []))
        init_message = getattr(session, "init_message", "")
        if (
            history
            and init_message
            and history[0].get("role") == "assistant"
            and history[0].get("content") == init_message
        ):
            history = history[1:]

        lines = ["# Короткий диалог", ""]
        user_turn = 0
        for item in history:
            role = item.get("role", "unknown")
            content = str(item.get("content", ""))
            if role == "user":
                user_turn += 1
                title = f"Пользователь, шаг {user_turn}"
            else:
                title = f"Агент, шаг {user_turn or 0}"
            lines.extend([f"## {title}", "", "```text", content, "```", ""])
        return "\n".join(lines).strip() + "\n"

    def _duplicate_metrics_markdown(
        self,
        *,
        first_text: str,
        second_text: str,
        ratio: float,
        min_ratio: float,
        max_ratio: float,
        detected_forbidden: list[str],
    ) -> str:
        lines = [
            "# Сравнение первого и повторного ответа",
            "",
            f"- Допустимый диапазон ratio: {min_ratio:.2f} .. {max_ratio:.2f}",
            "",
            "| Показатель | Значение |",
            "|---|---|",
            f"| Длина первого ответа | {len(first_text)} |",
            f"| Длина второго ответа | {len(second_text)} |",
            f"| Ratio длины | {ratio:.2f} |",
            f"| Технические артефакты | {', '.join(detected_forbidden) if detected_forbidden else 'нет'} |",
            f"| Превью первого ответа | {(self._preview_text(first_text) or '-').replace('|', '/')} |",
            f"| Превью второго ответа | {(self._preview_text(second_text) or '-').replace('|', '/')} |",
        ]
        return "\n".join(lines) + "\n"

    def _attach_case_report(
        self,
        *,
        title: str,
        scenario: str,
        expected: list[str],
        actual: list[str],
        verdict_title: str,
        verdict_text: str,
        findings: list[str],
        markdown_attachments: list[tuple[str, str]],
        technical: dict[str, Any],
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
        for name, content in markdown_attachments:
            attach_markdown(name, content)
        if findings:
            attach_text("findings.txt", "\n".join(f"- {item}" for item in findings))
        attach_json("[tech] reproducibility-case.json", technical)

    @pytest.mark.slow
    def test_reproducibility_jaccard(self, agent_client):
        """Independent runs of the same prompt yield sufficient lexical overlap. / Независимые запуски одного промпта дают достаточное лексическое совпадение."""
        cfg = get_config().reproducibility
        prompt = "Hello, I would like to get started"
        runs: list[dict[str, Any]] = []
        pairs: list[dict[str, Any]] = []

        try:
            with step("Сбор ответов в нескольких независимых запусках"):
                runs = self._collect_runs(
                    agent_client,
                    prompt=prompt,
                    runs_count=cfg.runs,
                )

            with step("Сравнение ответов по попарному Jaccard"):
                pairs = self._pairwise_jaccard(runs)
                for pair in pairs:
                    assert pair["jaccard"] >= cfg.jaccard_min, (
                        f"Runs {pair['left']} and {pair['right']}: "
                        f"Jaccard={pair['jaccard']:.2f} < {cfg.jaccard_min}"
                    )
        except Exception as exc:
            verdict_title, verdict_text = self._classify_failure(exc, case="jaccard")
            min_jaccard = (
                min(pair["jaccard"] for pair in pairs)
                if pairs
                else None
            )
            self._attach_case_report(
                title="Независимые запуски дают лексически близкие ответы",
                scenario=(
                    "Один и тот же запрос отправляется в несколько независимых "
                    "сессий. Ответы должны оставаться достаточно похожими по "
                    "словарному составу."
                ),
                expected=[
                    f"Выполняется {cfg.runs} независимых запусков с одним и тем же запросом",
                    f"Попарный Jaccard между ответами не ниже {cfg.jaccard_min:.2f}",
                ],
                actual=[
                    f"Количество успешно собранных запусков: {len(runs)}",
                    (
                        f"Минимальный Jaccard: {min_jaccard:.2f}"
                        if min_jaccard is not None
                        else "Минимальный Jaccard: не рассчитан"
                    ),
                    f"Ошибка: {self._preview_text(str(exc), limit=220)}",
                ],
                verdict_title=verdict_title,
                verdict_text=verdict_text,
                findings=[
                    "Независимые ответы на один и тот же запрос оказались слишком различными или проверка завершилась ошибкой.",
                ],
                markdown_attachments=[
                    ("runs.md", self._runs_markdown(runs) if runs else "# Независимые запуски\n\nНедостаточно данных.\n"),
                    (
                        "metrics.md",
                        self._jaccard_metrics_markdown(
                            pairs=pairs,
                            threshold=cfg.jaccard_min,
                        )
                        if pairs
                        else "# Попарная лексическая близость\n\nНедостаточно данных.\n",
                    ),
                ],
                technical={
                    "case": "reproducibility_jaccard",
                    "prompt": prompt,
                    "runs": self._serializable_runs(runs),
                    "pairs": pairs,
                    "runs_count": cfg.runs,
                    "jaccard_min": cfg.jaccard_min,
                },
            )
            raise

        self._attach_case_report(
            title="Независимые запуски дают лексически близкие ответы",
            scenario=(
                "Один и тот же запрос отправляется в несколько независимых "
                "сессий. Ответы должны оставаться достаточно похожими по "
                "словарному составу."
            ),
            expected=[
                f"Выполняется {cfg.runs} независимых запусков с одним и тем же запросом",
                f"Попарный Jaccard между ответами не ниже {cfg.jaccard_min:.2f}",
            ],
            actual=[
                f"Количество собранных запусков: {len(runs)}",
                f"Минимальный Jaccard: {min(pair['jaccard'] for pair in pairs):.2f}",
            ],
            verdict_title="Проверка пройдена",
            verdict_text="Независимые запуски дают достаточно похожие ответы и не распадаются на несвязанные варианты.",
            findings=[
                "Ответы на один и тот же запрос остаются воспроизводимыми по лексике.",
            ],
            markdown_attachments=[
                ("runs.md", self._runs_markdown(runs)),
                ("metrics.md", self._jaccard_metrics_markdown(pairs=pairs, threshold=cfg.jaccard_min)),
            ],
            technical={
                "case": "reproducibility_jaccard",
                "prompt": prompt,
                "runs": self._serializable_runs(runs),
                "pairs": pairs,
                "runs_count": cfg.runs,
                "jaccard_min": cfg.jaccard_min,
            },
        )

    def test_duplicate_message_remains_stable(self, session):
        """Sending the same message twice does not cause visible degradation. / Повтор одного и того же сообщения не вызывает заметной деградации ответа."""
        cfg = get_config().reproducibility
        prompt = "Hello"
        forbidden = ("traceback", "exception", "stack trace", "internal server error")
        first_text = ""
        second_text = ""
        ratio = 1.0
        detected_forbidden: list[str] = []

        try:
            with step("Отправка первого сообщения"):
                session.send(prompt)
                session.expect_response_ok()
                first_text = session.last_text

            with step("Отправка повторного сообщения"):
                session.send(prompt)
                session.expect_response_ok()
                second_text = session.last_text

            with step("Проверка отсутствия технических артефактов"):
                session.expect_not_contains(*forbidden)

            with step("Проверка, что длина повторного ответа не деградировала"):
                first_len = len(first_text)
                second_len = len(second_text)
                ratio = second_len / first_len if first_len else 1.0
                assert cfg.duplicate_length_ratio_min <= ratio <= cfg.duplicate_length_ratio_max, (
                    "Second response length degraded unexpectedly after duplicate input: "
                    f"{first_len} -> {second_len} chars (ratio={ratio:.2f}, "
                    f"allowed [{cfg.duplicate_length_ratio_min}..{cfg.duplicate_length_ratio_max}])"
                )
        except Exception as exc:
            detected_forbidden = self._detected_forbidden(session, forbidden)
            verdict_title, verdict_text = self._classify_failure(
                exc,
                case="duplicate",
                detected_forbidden=detected_forbidden,
            )
            self._attach_case_report(
                title="Повторное сообщение не вызывает заметной деградации ответа",
                scenario=(
                    "Пользователь дважды отправляет одно и то же короткое сообщение. "
                    "Повторный ответ не должен резко деградировать по длине и не "
                    "должен содержать технические артефакты."
                ),
                expected=[
                    "Оба ответа корректны и доступны пользователю",
                    "Во втором ответе нет технических артефактов",
                    (
                        "Отношение длины второго ответа к первому находится "
                        f"в диапазоне {cfg.duplicate_length_ratio_min:.2f} .. "
                        f"{cfg.duplicate_length_ratio_max:.2f}"
                    ),
                ],
                actual=[
                    f"Текущий номер хода: {getattr(session, 'turn', '-')}",
                    f"Сообщений в диалоге: {len(getattr(session, 'history', []))}",
                    f"Длина первого ответа: {len(first_text)}",
                    f"Длина второго ответа: {len(second_text)}",
                    f"Ratio длины: {ratio:.2f}",
                    (
                        "Найдены технические артефакты: " + ", ".join(detected_forbidden)
                        if detected_forbidden
                        else "Найдены технические артефакты: нет"
                    ),
                    f"Ошибка: {self._preview_text(str(exc), limit=220)}",
                ],
                verdict_title=verdict_title,
                verdict_text=verdict_text,
                findings=[
                    "Повторный ответ оказался нестабильным по длине или показал технические артефакты.",
                ],
                markdown_attachments=[
                    ("dialogue.md", self._dialogue_markdown(session)),
                    (
                        "comparison.md",
                        self._duplicate_metrics_markdown(
                            first_text=first_text,
                            second_text=second_text,
                            ratio=ratio,
                            min_ratio=cfg.duplicate_length_ratio_min,
                            max_ratio=cfg.duplicate_length_ratio_max,
                            detected_forbidden=detected_forbidden,
                        ),
                    ),
                ],
                technical={
                    "case": "duplicate_message_remains_stable",
                    "prompt": prompt,
                    "first_text": first_text,
                    "second_text": second_text,
                    "duplicate_length_ratio_min": cfg.duplicate_length_ratio_min,
                    "duplicate_length_ratio_max": cfg.duplicate_length_ratio_max,
                    "ratio": ratio,
                    "detected_forbidden": detected_forbidden,
                    "turn": getattr(session, "turn", None),
                },
            )
            raise

        self._attach_case_report(
            title="Повторное сообщение не вызывает заметной деградации ответа",
            scenario=(
                "Пользователь дважды отправляет одно и то же короткое сообщение. "
                "Повторный ответ не должен резко деградировать по длине и не "
                "должен содержать технические артефакты."
            ),
            expected=[
                "Оба ответа корректны и доступны пользователю",
                "Во втором ответе нет технических артефактов",
                (
                    "Отношение длины второго ответа к первому находится "
                    f"в диапазоне {cfg.duplicate_length_ratio_min:.2f} .. "
                    f"{cfg.duplicate_length_ratio_max:.2f}"
                ),
            ],
            actual=[
                f"Текущий номер хода: {getattr(session, 'turn', '-')}",
                f"Сообщений в диалоге: {len(getattr(session, 'history', []))}",
                f"Длина первого ответа: {len(first_text)}",
                f"Длина второго ответа: {len(second_text)}",
                f"Ratio длины: {ratio:.2f}",
                "Найдены технические артефакты: нет",
            ],
            verdict_title="Проверка пройдена",
            verdict_text="Повторный запрос не приводит к заметной деградации ответа и не добавляет технических артефактов.",
            findings=[
                "Повтор того же сообщения даёт стабильный по размеру пользовательский ответ.",
            ],
            markdown_attachments=[
                ("dialogue.md", self._dialogue_markdown(session)),
                (
                    "comparison.md",
                    self._duplicate_metrics_markdown(
                        first_text=first_text,
                        second_text=second_text,
                        ratio=ratio,
                        min_ratio=cfg.duplicate_length_ratio_min,
                        max_ratio=cfg.duplicate_length_ratio_max,
                        detected_forbidden=[],
                    ),
                ),
            ],
            technical={
                "case": "duplicate_message_remains_stable",
                "prompt": prompt,
                "first_text": first_text,
                "second_text": second_text,
                "duplicate_length_ratio_min": cfg.duplicate_length_ratio_min,
                "duplicate_length_ratio_max": cfg.duplicate_length_ratio_max,
                "ratio": ratio,
                "detected_forbidden": [],
                "turn": getattr(session, "turn", None),
            },
        )
