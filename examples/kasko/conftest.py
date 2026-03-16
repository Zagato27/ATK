"""
conftest.py for KASKO agent tests.

Demonstrates the fixture pattern: session-scoped client + per-test session.
"""
import pytest

from agent_test_kit import GigaChatJudge

from .client import KaskoClient, DEFAULT_EPK_ID
from .session import KaskoSession, kasko_registry

import os

GIGACHAT_BASE_URL = os.getenv("GIGACHAT_BASE_URL", "")
GIGACHAT_MODEL = os.getenv("GIGACHAT_MODEL", "GigaChat-Max")
GIGACHAT_CERT_FILE = os.getenv("GIGACHAT_CERT_FILE", "")
GIGACHAT_KEY_FILE = os.getenv("GIGACHAT_KEY_FILE", "")


@pytest.fixture(scope="session")
def judge_llm() -> GigaChatJudge:
    return GigaChatJudge(
        model_name=GIGACHAT_MODEL,
        base_url=GIGACHAT_BASE_URL,
        cert_file=GIGACHAT_CERT_FILE,
        key_file=GIGACHAT_KEY_FILE,
    )


@pytest.fixture(scope="session")
def kasko_client() -> KaskoClient:
    return KaskoClient()


@pytest.fixture(scope="session")
def kasko_client_long() -> KaskoClient:
    return KaskoClient(timeout=180)


@pytest.fixture
def kasko(kasko_client: KaskoClient) -> KaskoSession:
    """Fresh KaskoSession for deterministic tests (no judge required)."""
    session = KaskoSession(
        client=kasko_client,
        registry=kasko_registry(),
    )
    session.reset(epk_id=DEFAULT_EPK_ID)
    return session


@pytest.fixture
def kasko_judge(kasko_client: KaskoClient, judge_llm: GigaChatJudge) -> KaskoSession:
    """Fresh KaskoSession with LLM judge for semantic checks."""
    session = KaskoSession(
        client=kasko_client,
        judge=judge_llm,
        registry=kasko_registry(),
    )
    session.reset(epk_id=DEFAULT_EPK_ID)
    return session


@pytest.fixture
def kasko_long(kasko_client_long: KaskoClient, judge_llm: GigaChatJudge) -> KaskoSession:
    """KaskoSession with extended timeout for E2E / long dialogues."""
    session = KaskoSession(
        client=kasko_client_long,
        judge=judge_llm,
        registry=kasko_registry(),
    )
    session.reset(epk_id=DEFAULT_EPK_ID)
    return session
