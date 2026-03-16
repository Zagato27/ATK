"""
KaskoSession — KASKO-specific extension of AgentSession.

Adds domain-specific ``expect_*`` methods and metrics.
"""
from __future__ import annotations

from agent_test_kit import AgentSession, MetricRegistry, default_registry

from .client import KaskoClient, InitData


# -- KASKO-specific metrics -------------------------------------------------

KASKO_METRICS: dict[str, str] = {
    "initialization_response": (
        "Оцени первое приветственное сообщение агента-страховщика клиенту.\n"
        "1. Агент поприветствовал пользователя по имени\n"
        "2. Агент пригласил к расчёту КАСКО или предложил помощь\n"
        "3. Ответ не содержит лишней информации\n"
        "4. Формат вежливый, профессиональный, на «Вы»"
    ),
    "tariff_presentation": (
        "Оцени полноту результата расчёта КАСКО.\n"
        "1. Указан размер страховой премии\n"
        "2. Указан размер страховой суммы\n"
        "3. Указан размер франшизы\n"
        "4. Указана противоугонная система\n"
        "5. Указана защитная плёнка\n"
        "6. Указан состав водителей"
    ),
    "driver_flow": (
        "Оцени, корректно ли агент ведёт сбор данных о водителях.\n"
        "1. Агент запросил ФИО, ДР, ВУ, стаж\n"
        "2. При >1 водителе уточнил основного\n"
        "3. При отказе предложил непоименованный / мультидрайв\n"
        "4. Каскад: поименованный → непоименованный → мультидрайв"
    ),
    "early_termination": (
        "Оцени, корректно ли агент обработал желание пользователя прекратить.\n"
        "1. Агент прекратил задавать вопросы\n"
        "2. Агент попытался рассчитать с имеющимися данными\n"
        "3. Агент остался вежливым\n"
        "4. Агент использовал стандартные значения для незаполненных полей"
    ),
}


def kasko_registry() -> MetricRegistry:
    """Registry with built-in + KASKO-specific metrics."""
    reg = default_registry()
    reg.register_bulk(KASKO_METRICS)
    return reg


# -- session ----------------------------------------------------------------

class KaskoSession(AgentSession):
    """AgentSession with KASKO-specific assertions."""

    @property
    def kasko_init(self) -> InitData:
        client = self._client
        assert isinstance(client, KaskoClient), "Client must be KaskoClient"
        assert client.last_init is not None, "Session not initialized"
        return client.last_init

    # -- init checks --------------------------------------------------------

    def expect_init_greets(self) -> KaskoSession:
        """Init message contains a greeting."""
        greetings = {
            "здравствуйте", "добрый день", "добрый вечер",
            "добро пожаловать", "приветствую", "рады", "рада",
        }
        text = self.init_message.lower()
        assert any(g in text for g in greetings), (
            f"Init message has no greeting: '{self.init_message}'"
        )
        return self

    def expect_init_mentions_user(self) -> KaskoSession:
        """Init message mentions the user's name."""
        name_parts = self.kasko_init.user_name.split()
        msg = self.init_message.lower()
        found = any(part.lower() in msg for part in name_parts if len(part) > 2)
        assert found, (
            f"Greeting does not mention '{self.kasko_init.user_name}': "
            f"'{self.init_message}'"
        )
        return self

    def expect_vehicles_found(self, min_count: int = 1) -> KaskoSession:
        assert len(self.kasko_init.vehicles) >= min_count, (
            f"Expected >= {min_count} vehicles, got {len(self.kasko_init.vehicles)}"
        )
        return self

    # -- chat checks --------------------------------------------------------

    def expect_stays_on_topic(self) -> KaskoSession:
        """Response is related to KASKO / insurance."""
        self._need_response()
        terms = {"каско", "страхов", "полис", "тариф", "премия", "vin", "вин"}
        text = self.last_response.text.lower()
        assert any(t in text for t in terms), (
            f"Turn {self.turn}: off-topic response.\n"
            f"Response: '{self.last_text[:300]}'"
        )
        return self

    def expect_tariff_result(self, *fields: str) -> KaskoSession:
        """Response contains tariff components."""
        self._need_response()
        tariff_keywords: dict[str, set[str]] = {
            "премия": {"преми", "стоимость полиса", "к оплате"},
            "сумма": {"страховая сумма", "сумма"},
            "франшиза": {"франшиз"},
            "водители": {"водител", "допущен"},
            "противоугонка": {"сигнализ", "противоугон", "охран"},
        }
        text = self.last_response.text.lower()
        check = fields or tuple(tariff_keywords.keys())
        missing = []
        for f in check:
            synonyms = tariff_keywords.get(f, {f})
            if not any(s in text for s in synonyms):
                missing.append(f)
        assert not missing, (
            f"Turn {self.turn}: tariff missing: {missing}\n"
            f"Response: '{self.last_text[:500]}'"
        )
        return self

    def expect_tool_called(self, tool_name: str) -> KaskoSession:
        """Assert a specific tool was invoked."""
        self._need_response()
        tool_calls = self.last_response.metadata.get("tool_calls", [])
        names = [
            tc.get("name", tc.get("function", ""))
            for tc in tool_calls
        ]
        assert any(tool_name.lower() in n.lower() for n in names), (
            f"Turn {self.turn}: expected tool '{tool_name}', called: {names}"
        )
        return self

    def expect_node(self, expected: str) -> KaskoSession:
        """Assert which agent node handled the request."""
        self._need_response()
        actual = self.last_response.metadata.get("node", "")
        assert actual == expected, (
            f"Turn {self.turn}: node='{actual}', expected '{expected}'"
        )
        return self
