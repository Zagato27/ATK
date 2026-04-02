"""Unit tests for optional Allure helpers."""
from agent_test_kit.allure_support import sanitize_for_allure


def test_sanitize_for_allure_redacts_sensitive_keys_and_pii():
    payload = {
        "api_key": "super-secret",
        "note": "email user@example.com phone +1 202 555 0199 card 4111 1111 1111 1111",
        "nested": {
            "token": "abc123",
            "text": "Bearer very-secret-token",
        },
    }

    sanitized = sanitize_for_allure(payload)

    assert sanitized["api_key"] == "[REDACTED]"
    assert sanitized["nested"]["token"] == "[REDACTED]"
    assert "[REDACTED_EMAIL]" in sanitized["note"]
    assert "[REDACTED_PHONE]" in sanitized["note"]
    assert "[REDACTED_CARD]" in sanitized["note"]
    assert "Bearer [REDACTED]" in sanitized["nested"]["text"]
