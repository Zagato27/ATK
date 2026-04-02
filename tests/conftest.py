"""Shared fixtures for framework unit tests."""
import logging
import os
import sys
from pathlib import Path

import pytest

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover - optional in CI/local envs
    def load_dotenv(*args, **kwargs):
        return False

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
os.environ.setdefault("ATK_CONFIG_PATH", str(ROOT_DIR / "agent-test-kit.toml"))

from agent_test_kit import (
    AgentResponse,
    BaseAgentClient,
    AgentSession,
    ConfiguredAgentClient,
    create_judge_from_config,
    get_config,
)

logger = logging.getLogger(__name__)


class MockClient(BaseAgentClient):
    """In-memory mock client that returns predefined responses."""

    def __init__(self, responses: list[str] | None = None):
        super().__init__(base_url="http://mock", timeout=1, verify=False)
        self._responses = list(responses or ["Mock reply"])
        self._call_idx = 0
        self._init_data = {
            "session_id": "mock-session-1",
            "message": "Hello! How can I help you?",
            "user_name": "Test User",
        }

    def create_session(self, **kwargs):
        self.session_id = self._init_data["session_id"]
        return dict(self._init_data)

    def send_message(self, message, **kwargs):
        text = (
            self._responses[self._call_idx]
            if self._call_idx < len(self._responses)
            else self._responses[-1]
        )
        self._call_idx += 1
        return AgentResponse(text=text, status_code=200, raw={"response": text})

    def reset(self):
        super().reset()
        self._call_idx = 0


@pytest.fixture
def mock_client():
    return MockClient()


# ======================================================================
# Config-driven fixtures for generic/integration tests
# ======================================================================

@pytest.fixture(scope="session")
def agent_client():
    """HTTP client for the tested agent service configured in [agent]."""
    cfg = get_config().agent
    logger.info(
        "agent_client: %s (init=%s, chat=%s)",
        cfg.base_url,
        cfg.init_path,
        cfg.chat_path,
    )
    return ConfiguredAgentClient(cfg)


@pytest.fixture(scope="session")
def judge_llm():
    """Real judge built from config. No mock fallback is used."""
    cfg = get_config()
    provider = cfg.judge.provider.lower()
    has_direct_key = bool(cfg.judge.api_key)
    has_env_key = bool(cfg.judge.api_key_env and os.getenv(cfg.judge.api_key_env))
    has_cert_pair = bool(cfg.judge.cert_file and cfg.judge.key_file)
    has_env_cert_pair = bool(
        cfg.judge.cert_file_env and os.getenv(cfg.judge.cert_file_env)
        and cfg.judge.key_file_env and os.getenv(cfg.judge.key_file_env)
    )

    if provider == "gigachat":
        if not (has_cert_pair or has_env_cert_pair):
            raise RuntimeError(
                "GigaChat judge is not configured. Set [judge].cert_file/[judge].key_file "
                "or [judge].cert_file_env/[judge].key_file_env in agent-test-kit.toml."
            )
    elif not (has_direct_key or has_env_key):
        raise RuntimeError(
            "Judge is not configured. Set [judge].api_key or [judge].api_key_env "
            "in agent-test-kit.toml."
        )

    judge = create_judge_from_config(cfg.judge)
    logger.info(
        "judge_llm: REAL %s → %s (%s)",
        cfg.judge.provider,
        cfg.judge.model_name,
        cfg.judge.api_base_url or "default endpoint",
    )
    return judge


@pytest.fixture
def session(agent_client):
    """AgentSession wired to the configured agent service."""
    s = AgentSession(client=agent_client)
    s.init_session()
    return s
