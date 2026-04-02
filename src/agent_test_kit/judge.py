"""
LLM-as-Judge adapters.

* :class:`BaseLLMJudge` — abstract base for judge implementations.
  When ``deepeval`` is installed, also extends ``DeepEvalBaseLLM``
  so judges can be passed directly to DeepEval's ``GEval`` metric.
* :class:`GigaChatJudge` — GigaChat via mTLS (extra ``gigachat``).
* :class:`OpenAIJudge` — OpenAI-compatible APIs (extra ``openai``).
* :class:`AnthropicJudge` — Anthropic Claude API (extra ``anthropic``).
* :func:`create_judge_from_config` — factory that builds the right judge
  from ``[judge]`` section of ``agent-test-kit.toml``.

Адаптеры LLM-as-Judge. Базовый класс BaseLLMJudge — абстрактная база;
при установленном deepeval наследует DeepEvalBaseLLM для совместимости.
"""
from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod

try:
    from deepeval.models import DeepEvalBaseLLM
    _DEEPEVAL_AVAILABLE = True
except ImportError:
    _DEEPEVAL_AVAILABLE = False

logger = logging.getLogger(__name__)

_JudgeBase = DeepEvalBaseLLM if _DEEPEVAL_AVAILABLE else ABC


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseLLMJudge(_JudgeBase):  # type: ignore[misc]
    """Extend this class to plug in any LLM as a judge for ``evaluate()``.

    Расширьте этот класс, чтобы подключить любой LLM в качестве судьи для ``evaluate()``.
    """

    @abstractmethod
    def generate(self, prompt: str) -> str:  # pragma: no cover
        ...

    async def a_generate(self, prompt: str) -> str:
        """Async wrapper for generate; defaults to sync. / Асинхронная обёртка для generate; по умолчанию синхронная."""
        return self.generate(prompt)

    def load_model(self):
        """Return model name for DeepEval compatibility. / Возвращает имя модели для совместимости с DeepEval."""
        return self.get_model_name()

    @abstractmethod
    def get_model_name(self) -> str:  # pragma: no cover
        ...


# ---------------------------------------------------------------------------
# GigaChat implementation (optional dependency)
# ---------------------------------------------------------------------------

class GigaChatJudge(BaseLLMJudge):
    """GigaChat judge connected via mTLS client certificate.

    Requires ``langchain-gigachat`` and ``langchain-core`` —
    install with ``pip install agent-test-kit[gigachat]``.

    Судья GigaChat, подключаемый через mTLS клиентский сертификат. Требует
    ``langchain-gigachat`` и ``langchain-core`` — установка: ``pip install agent-test-kit[gigachat]``.
    """

    def __init__(
        self,
        model_name: str = "GigaChat-Max",
        base_url: str = "",
        cert_file: str = "",
        key_file: str = "",
        verify_ssl: bool = False,
        timeout: int = 120,
    ):
        try:
            from langchain_gigachat.chat_models import GigaChat as LC_GigaChat
        except ImportError as exc:
            raise ImportError(
                "GigaChatJudge requires langchain-gigachat. "
                "Install with: pip install agent-test-kit[gigachat]"
            ) from exc

        cert_file = os.path.abspath(cert_file)
        key_file = os.path.abspath(key_file)

        if not os.path.isfile(cert_file):
            raise FileNotFoundError(f"GigaChat certificate not found: {cert_file}")
        if not os.path.isfile(key_file):
            raise FileNotFoundError(f"GigaChat private key not found: {key_file}")

        self.model_name = model_name
        self._client = LC_GigaChat(
            model=model_name,
            base_url=base_url,
            cert_file=cert_file,
            key_file=key_file,
            verify_ssl_certs=verify_ssl,
            timeout=timeout,
        )
        logger.info(
            "GigaChatJudge ready: model=%s url=%s cert=%s",
            model_name, base_url, cert_file,
        )

    def generate(self, prompt: str) -> str:
        """Generate judge response via GigaChat. / Генерирует ответ судьи через GigaChat."""
        try:
            from langchain_core.messages import HumanMessage
        except ImportError:
            from langchain.schema import HumanMessage  # type: ignore[no-redef]

        logger.debug(">>> GigaChat judge prompt (%d chars)", len(prompt))
        response = self._client.invoke([HumanMessage(content=prompt)])
        logger.debug("<<< GigaChat judge response (%d chars)", len(response.content))
        return response.content

    def get_model_name(self) -> str:
        return self.model_name


# ---------------------------------------------------------------------------
# OpenAI / OpenAI-compatible implementation (optional dependency)
# ---------------------------------------------------------------------------

class OpenAIJudge(BaseLLMJudge):
    """Judge powered by the OpenAI API (or any OpenAI-compatible endpoint).

    Works with OpenAI, Azure OpenAI, LiteLLM proxy, vLLM, Ollama, etc. —
    anything that speaks the ``openai`` Python SDK protocol.

    Install with ``pip install agent-test-kit[openai]``.

    Parameters:
        model_name: Model identifier, e.g. ``"gpt-4o"``, ``"gpt-4.1-mini"``.
        api_key:    API key.  Falls back to ``OPENAI_API_KEY`` env var.
        base_url:   Custom base URL (for proxies / self-hosted).
                    Falls back to ``OPENAI_BASE_URL`` env var, then the
                    default ``https://api.openai.com/v1``.
        temperature: Sampling temperature (0 for deterministic judge).
        timeout:    Request timeout in seconds.

    Судья на базе OpenAI API (или любого OpenAI-совместимого эндпоинта).
    Работает с OpenAI, Azure OpenAI, LiteLLM, vLLM, Ollama и т.д.
    """

    def __init__(
        self,
        model_name: str = "gpt-4o",
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.0,
        timeout: int = 120,
    ):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "OpenAIJudge requires the openai package. "
                "Install with: pip install agent-test-kit[openai]"
            ) from exc

        resolved_key = api_key or os.getenv("OPENAI_API_KEY", "")
        resolved_url = base_url or os.getenv("OPENAI_BASE_URL")

        if not resolved_key:
            raise ValueError(
                "OpenAI API key not provided. "
                "Pass api_key= or set OPENAI_API_KEY env var."
            )

        self.model_name = model_name
        self._temperature = temperature
        self._client = OpenAI(
            api_key=resolved_key,
            base_url=resolved_url,
            timeout=timeout,
        )
        logger.info(
            "OpenAIJudge ready: model=%s base_url=%s",
            model_name, resolved_url or "default",
        )

    def generate(self, prompt: str) -> str:
        """Generate judge response via OpenAI API. / Генерирует ответ судьи через OpenAI API."""
        logger.debug(">>> OpenAI judge prompt (%d chars)", len(prompt))
        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self._temperature,
        )
        text = response.choices[0].message.content or ""
        logger.debug("<<< OpenAI judge response (%d chars)", len(text))
        return text

    def get_model_name(self) -> str:
        return self.model_name


# ---------------------------------------------------------------------------
# Anthropic implementation (optional dependency)
# ---------------------------------------------------------------------------

class AnthropicJudge(BaseLLMJudge):
    """Judge powered by the Anthropic Claude API.

    Install with ``pip install agent-test-kit[anthropic]``.

    Parameters:
        model_name:  Model identifier, e.g. ``"claude-sonnet-4-20250514"``.
        api_key:     API key.  Falls back to ``ANTHROPIC_API_KEY`` env var.
        base_url:    Custom base URL (for proxies).
                     Falls back to ``ANTHROPIC_BASE_URL`` env var.
        temperature: Sampling temperature (0 for deterministic judge).
        max_tokens:  Maximum tokens in the judge response.
        timeout:     Request timeout in seconds.

    Судья на базе Anthropic Claude API. Установка: ``pip install agent-test-kit[anthropic]``.
    """

    def __init__(
        self,
        model_name: str = "claude-sonnet-4-20250514",
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        timeout: int = 120,
    ):
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError(
                "AnthropicJudge requires the anthropic package. "
                "Install with: pip install agent-test-kit[anthropic]"
            ) from exc

        resolved_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        resolved_url = base_url or os.getenv("ANTHROPIC_BASE_URL")

        if not resolved_key:
            raise ValueError(
                "Anthropic API key not provided. "
                "Pass api_key= or set ANTHROPIC_API_KEY env var."
            )

        self.model_name = model_name
        self._temperature = temperature
        self._max_tokens = max_tokens

        kwargs: dict = {"api_key": resolved_key, "timeout": timeout}
        if resolved_url:
            kwargs["base_url"] = resolved_url

        self._client = anthropic.Anthropic(**kwargs)
        logger.info(
            "AnthropicJudge ready: model=%s base_url=%s",
            model_name, resolved_url or "default",
        )

    def generate(self, prompt: str) -> str:
        """Generate judge response via Anthropic API. / Генерирует ответ судьи через Anthropic API."""
        logger.debug(">>> Anthropic judge prompt (%d chars)", len(prompt))
        response = self._client.messages.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        texts: list[str] = []
        block_types: list[str] = []
        for block in getattr(response, "content", []):
            if isinstance(block, dict):
                block_types.append(str(block.get("type", "dict")))
                block_text = block.get("text")
            else:
                block_types.append(type(block).__name__)
                block_text = getattr(block, "text", None)
            if isinstance(block_text, str) and block_text.strip():
                texts.append(block_text)

        if not texts:
            raise ValueError(
                "Anthropic response has no text blocks. "
                f"Block types: {block_types}"
            )

        text = "\n".join(texts)
        logger.debug("<<< Anthropic judge response (%d chars)", len(text))
        return text

    def get_model_name(self) -> str:
        return self.model_name


# ---------------------------------------------------------------------------
# Factory: build judge from config
# ---------------------------------------------------------------------------

def create_judge_from_config(
    judge_cfg: "JudgeConfig | None" = None,
) -> BaseLLMJudge:
    """Instantiate the right judge adapter from ``[judge]`` config section.

    Reads ``provider``, ``api_base_url``, ``model_name``, ``api_key_env``
    (env-var name), ``judge_temperature``, ``max_tokens``, ``timeout``
    from :class:`JudgeConfig`.

    Создаёт нужный judge-адаптер по секции ``[judge]`` конфига.
    """
    if judge_cfg is None:
        from agent_test_kit.config import get_config
        judge_cfg = get_config().judge

    provider = judge_cfg.provider.lower()
    base_url = judge_cfg.api_base_url or None
    model = judge_cfg.model_name

    if provider == "gigachat":
        cert_file = judge_cfg.cert_file or (
            os.getenv(judge_cfg.cert_file_env) if judge_cfg.cert_file_env else ""
        )
        key_file = judge_cfg.key_file or (
            os.getenv(judge_cfg.key_file_env) if judge_cfg.key_file_env else ""
        )
        if not cert_file or not key_file:
            raise ValueError(
                "GigaChatJudge requires cert_file and key_file in [judge] config. "
                "Set judge.cert_file/judge.key_file or "
                "judge.cert_file_env/judge.key_file_env in agent-test-kit.toml."
            )
        return GigaChatJudge(
            model_name=model or "GigaChat-Max",
            base_url=base_url or "",
            cert_file=cert_file,
            key_file=key_file,
            verify_ssl=judge_cfg.verify_ssl,
            timeout=judge_cfg.timeout,
        )

    api_key = judge_cfg.api_key or None
    if not api_key and judge_cfg.api_key_env:
        api_key = os.getenv(judge_cfg.api_key_env)

    if not api_key:
        hint = (
            f"Set judge.api_key in agent-test-kit.toml, "
            f"or set env var ${judge_cfg.api_key_env or 'ANTHROPIC_API_KEY'}"
        )
        raise ValueError(f"Judge API key not found. {hint}")

    if provider == "anthropic":
        return AnthropicJudge(
            model_name=model or "claude-sonnet-4-20250514",
            api_key=api_key,
            base_url=base_url,
            temperature=judge_cfg.judge_temperature,
            max_tokens=judge_cfg.max_tokens,
            timeout=judge_cfg.timeout,
        )

    if provider == "openai":
        return OpenAIJudge(
            model_name=model or "gpt-4o",
            api_key=api_key,
            base_url=base_url,
            temperature=judge_cfg.judge_temperature,
            timeout=judge_cfg.timeout,
        )

    raise ValueError(
        f"Unsupported judge provider: '{provider}'. "
        f"Supported: anthropic, openai, gigachat."
    )
