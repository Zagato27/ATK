"""Shared fixtures for framework unit tests."""
import pytest

from agent_test_kit import AgentResponse, BaseAgentClient, AgentSession


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


@pytest.fixture
def session(mock_client):
    s = AgentSession(client=mock_client)
    s.init_session()
    return s
