"""
agent-test-kit — reusable testing framework for LLM agents.

Quick start::

    from agent_test_kit import AgentSession, BaseAgentClient, run_dialogue  # noqa: F811

Generic test suites::

    from agent_test_kit.generic_tests import GenericEdgeCaseTests, GenericPromptSecurityTests

agent-test-kit — переиспользуемый фреймворк для тестирования LLM-агентов.
Быстрый старт и общие тестовые наборы доступны через импорт основных классов.
"""
from agent_test_kit.response import AgentResponse
from agent_test_kit.client import BaseAgentClient
from agent_test_kit.http_client import ConfiguredAgentClient
from agent_test_kit.config import Config, get_config, reload_config
from agent_test_kit.session import AgentSession
from agent_test_kit.judge import (
    BaseLLMJudge,
    GigaChatJudge,
    OpenAIJudge,
    AnthropicJudge,
    create_judge_from_config,
)
from agent_test_kit.geval import ATKGEval, GEvalResult
from agent_test_kit.golden import (
    GoldenCase,
    GoldenReport,
    DriftItem,
    load_golden,
    save_golden,
    compare_run,
    text_hash,
)
from agent_test_kit.metrics import MetricRegistry, default_registry, BUILTIN_METRICS
from agent_test_kit.statistical import Distribution, RunResult, run_n_times, mann_whitney_u
from agent_test_kit.session import run_dialogue

__all__ = [
    "AgentResponse",
    "BaseAgentClient",
    "ConfiguredAgentClient",
    "Config",
    "get_config",
    "reload_config",
    "AgentSession",
    "BaseLLMJudge",
    "GigaChatJudge",
    "OpenAIJudge",
    "AnthropicJudge",
    "create_judge_from_config",
    "MetricRegistry",
    "default_registry",
    "ATKGEval",
    "GEvalResult",
    "GoldenCase",
    "GoldenReport",
    "DriftItem",
    "load_golden",
    "save_golden",
    "compare_run",
    "text_hash",
    "BUILTIN_METRICS",
    "Distribution",
    "RunResult",
    "run_n_times",
    "mann_whitney_u",
    "run_dialogue",
]
