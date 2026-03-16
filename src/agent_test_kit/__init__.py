"""
agent-test-kit — reusable testing framework for LLM agents.

Quick start::

    from agent_test_kit import AgentSession, BaseAgentClient, run_dialogue

Generic test suites::

    from agent_test_kit.generic_tests import GenericEdgeCaseTests, GenericSecurityTests

agent-test-kit — переиспользуемый фреймворк для тестирования LLM-агентов.
Быстрый старт и общие тестовые наборы доступны через импорт основных классов.
"""
from agent_test_kit.response import AgentResponse
from agent_test_kit.client import BaseAgentClient
from agent_test_kit.config import Config, get_config, reload_config
from agent_test_kit.session import AgentSession
from agent_test_kit.judge import (
    BaseLLMJudge,
    GigaChatJudge,
    OpenAIJudge,
    AnthropicJudge,
)
from agent_test_kit.metrics import MetricRegistry, default_registry, BUILTIN_METRICS
from agent_test_kit.helpers import run_dialogue

__all__ = [
    "AgentResponse",
    "BaseAgentClient",
    "Config",
    "get_config",
    "reload_config",
    "AgentSession",
    "BaseLLMJudge",
    "GigaChatJudge",
    "OpenAIJudge",
    "AnthropicJudge",
    "MetricRegistry",
    "default_registry",
    "BUILTIN_METRICS",
    "run_dialogue",
]
