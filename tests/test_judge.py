"""Unit tests for judge adapters."""

import pytest

import agent_test_kit.judge as judge_module
from agent_test_kit.config import JudgeConfig
from agent_test_kit.judge import AnthropicJudge, create_judge_from_config


class _DummyMessages:
    def __init__(self, response):
        self._response = response

    def create(self, **kwargs):  # noqa: ANN003
        return self._response


class _DummyClient:
    def __init__(self, response):
        self.messages = _DummyMessages(response)


class _Response:
    def __init__(self, content):
        self.content = content


class _TextBlock:
    def __init__(self, text: str):
        self.text = text


class _ThinkingBlock:
    pass


def _make_judge_with_response(response: _Response) -> AnthropicJudge:
    judge = AnthropicJudge.__new__(AnthropicJudge)
    judge.model_name = "mock-model"
    judge._temperature = 0.0
    judge._max_tokens = 128
    judge._client = _DummyClient(response)
    return judge


class TestAnthropicJudgeGenerate:
    def test_extracts_text_block(self):
        judge = _make_judge_with_response(
            _Response([_TextBlock("ok")])
        )
        assert judge.generate("ping") == "ok"

    def test_skips_thinking_block_and_uses_text(self):
        judge = _make_judge_with_response(
            _Response([_ThinkingBlock(), _TextBlock("answer")])
        )
        assert judge.generate("ping") == "answer"

    def test_supports_dict_blocks(self):
        judge = _make_judge_with_response(
            _Response([{"type": "text", "text": "dict answer"}])
        )
        assert judge.generate("ping") == "dict answer"

    def test_joins_multiple_text_blocks(self):
        judge = _make_judge_with_response(
            _Response([_TextBlock("line1"), _TextBlock("line2")])
        )
        assert judge.generate("ping") == "line1\nline2"

    def test_raises_when_no_text_blocks(self):
        judge = _make_judge_with_response(
            _Response([_ThinkingBlock(), {"type": "thinking"}])
        )
        with pytest.raises(ValueError, match="no text blocks"):
            judge.generate("ping")


class TestCreateJudgeFromConfig:
    def test_gigachat_uses_env_paths_for_certificates(self, monkeypatch):
        captured: dict[str, object] = {}

        class _StubGigaChatJudge:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        monkeypatch.setenv("GIGACHAT_CERT_FILE", r"C:\certs\client.pem")
        monkeypatch.setenv("GIGACHAT_KEY_FILE", r"C:\certs\client.key")
        monkeypatch.setattr(judge_module, "GigaChatJudge", _StubGigaChatJudge)

        cfg = JudgeConfig(
            provider="gigachat",
            model_name="GigaChat-Max",
            api_base_url="https://gigachat.devices.sberbank.ru/api/v1",
            cert_file_env="GIGACHAT_CERT_FILE",
            key_file_env="GIGACHAT_KEY_FILE",
            verify_ssl=True,
            timeout=90,
        )

        judge = create_judge_from_config(cfg)

        assert isinstance(judge, _StubGigaChatJudge)
        assert captured == {
            "model_name": "GigaChat-Max",
            "base_url": "https://gigachat.devices.sberbank.ru/api/v1",
            "cert_file": r"C:\certs\client.pem",
            "key_file": r"C:\certs\client.key",
            "verify_ssl": True,
            "timeout": 90,
        }

    def test_gigachat_requires_certificates_or_envs(self):
        cfg = JudgeConfig(provider="gigachat")

        with pytest.raises(ValueError, match="cert_file"):
            create_judge_from_config(cfg)
