from __future__ import annotations

from bedrock_client import _is_throttling_status


def test_429_is_throttling_even_with_context_overflow_text() -> None:
    """Regression test: a 429 whose body happens to mention 'input tokens' must
    still be retried as throttling, not treated as a deterministic overflow."""
    body = "Too many requests. Input is too long: contains at least 300000 input tokens"
    assert _is_throttling_status(429, body) is True


def test_non_429_context_overflow_is_not_throttling() -> None:
    body = "Input is too long: contains at least 300000 input tokens"
    assert _is_throttling_status(400, body) is False


def test_429_with_no_body_is_throttling() -> None:
    assert _is_throttling_status(429, "") is True


def test_5xx_with_throttle_text_is_throttling() -> None:
    assert _is_throttling_status(503, "on-demand capacity exceeded") is True


def test_400_with_no_matching_text_is_not_throttling() -> None:
    assert _is_throttling_status(400, "some other client error") is False


def test_200_is_not_throttling() -> None:
    assert _is_throttling_status(200, "") is False
