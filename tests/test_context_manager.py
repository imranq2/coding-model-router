"""Tests for context_manager.py. Tokenizer is mocked — no network calls."""
from __future__ import annotations

from typing import Any, Sequence
from unittest.mock import patch

from context_manager import (
    _TOOL_COMPRESS_THRESHOLD_CHARS,
    _TOOL_HEAD_CHARS,
    _TOOL_TAIL_CHARS,
    _TRUNCATION_MARKER,
    ContextBudget,
    _apply_output_budget_cap,
    build_budget,
    compress_tool_result_text,
    enforce_context_budget,
)

FAKE_MODEL = "test-tokenizer/does-not-exist"


def _msg(role: str, content: str = "", **kw: object) -> dict[str, Any]:
    return {"role": role, "content": content, **kw}


def _tool_call_group(tool_call_id: str, result: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": tool_call_id, "type": "function", "function": {"name": "bash", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": tool_call_id, "content": result},
    ]


def _body(messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None, max_tokens: int = 8192) -> dict[str, Any]:
    b: dict[str, Any] = {"model": "m", "messages": messages, "max_tokens": max_tokens}
    if tools:
        b["tools"] = tools
    return b


def _mock_count(value: int | None):
    return patch("context_manager.count_oai_request_tokens", return_value=value)


def _mock_count_seq(values: Sequence[int | None]):
    return patch("context_manager.count_oai_request_tokens", side_effect=values)


def test_budget_defaults() -> None:
    b = ContextBudget()
    assert b.backend_max_context_tokens == 262144
    assert b.reserved_output_tokens == 16384
    assert b.tokenizer_safety_margin == 6000
    assert b.effective_input_tokens == 262144 - 16384 - 6000


def test_build_budget_from_route() -> None:
    route = {"backend_max_context_tokens": 200000, "reserved_output_tokens": 20000, "tokenizer_safety_margin": 5000}
    b = build_budget(route)
    assert b.effective_input_tokens == 175000


def test_build_budget_explicit_effective_input_tokens() -> None:
    b = build_budget({"effective_input_tokens": 240000})
    assert b.effective_input_tokens == 240000


def test_build_budget_empty_route_uses_defaults() -> None:
    assert build_budget({}) == ContextBudget()


def test_compress_short_text_unchanged() -> None:
    text = "short result"
    assert compress_tool_result_text(text) == text


def test_compress_long_text_structure() -> None:
    head = "A" * _TOOL_HEAD_CHARS
    middle = "B" * 10000
    tail = "C" * _TOOL_TAIL_CHARS
    result = compress_tool_result_text(head + middle + tail)
    assert result.startswith(head)
    assert result.endswith(tail)
    assert _TRUNCATION_MARKER in result


def test_compress_tail_preserved_for_stack_traces() -> None:
    noise = "INFO: compiling...\n" * 500
    error_line = "FATAL: test_foo: AssertionError('expected 42, got 0')"
    compressed = compress_tool_result_text(noise + error_line)
    assert error_line in compressed


def test_compress_tail_larger_than_head() -> None:
    assert _TOOL_TAIL_CHARS > _TOOL_HEAD_CHARS


def test_no_compression_when_under_budget() -> None:
    messages = [_msg("system", "sys"), _msg("user", "hello")]
    body = _body(messages)
    with _mock_count(1000):
        result = enforce_context_budget(body, {}, FAKE_MODEL)
    assert result["messages"] == messages


def test_tokenizer_unavailable_small_body_passes_through() -> None:
    body = _body([_msg("user", "hello")])
    with _mock_count(None):
        result = enforce_context_budget(body, {}, FAKE_MODEL)
    assert result is body


def test_char_estimate_fallback_compresses_when_over_budget() -> None:
    route = {"backend_max_context_tokens": 10000, "reserved_output_tokens": 3000, "tokenizer_safety_margin": 5000}
    big_tool = "X" * 100_000
    messages = [_msg("system", "sys"), *_tool_call_group("tc1", big_tool), _msg("user", "CURRENT")]
    body = _body(messages)
    with _mock_count(None):
        result = enforce_context_budget(body, route, FAKE_MODEL)
    tool_msg = next(m for m in result["messages"] if m.get("role") == "tool")
    assert _TRUNCATION_MARKER in tool_msg["content"]


def test_oversized_tool_result_compressed() -> None:
    big = "Z" * (2 * _TOOL_COMPRESS_THRESHOLD_CHARS)
    messages = [_msg("system", "sys"), *_tool_call_group("tc1", big), _msg("user", "continue")]
    body = _body(messages)
    with _mock_count_seq([300000, 200000]):
        result = enforce_context_budget(body, {}, FAKE_MODEL)
    tool_msg = next(m for m in result["messages"] if m.get("role") == "tool")
    assert _TRUNCATION_MARKER in tool_msg["content"]
    assert len(tool_msg["content"]) < len(big)


def test_small_tool_result_not_compressed() -> None:
    small = "output: ok"
    messages = [*_tool_call_group("tc1", small), _msg("user", "done")]
    body = _body(messages)
    with _mock_count(1000):
        result = enforce_context_budget(body, {}, FAKE_MODEL)
    tool_msg = next(m for m in result["messages"] if m.get("role") == "tool")
    assert tool_msg["content"] == small


def test_drops_oldest_groups_when_compression_insufficient() -> None:
    messages = [_msg("system", "sys"), _msg("user", "old question"), _msg("assistant", "old answer"), _msg("user", "CURRENT REQUEST")]
    body = _body(messages)
    with _mock_count_seq([300000, 250000, 200000]):
        result = enforce_context_budget(body, {}, FAKE_MODEL)
    contents = [m.get("content") for m in result["messages"]]
    assert "CURRENT REQUEST" in contents
    assert "old answer" not in contents


def test_system_message_never_dropped() -> None:
    groups: list[dict[str, Any]] = []
    for i in range(10):
        groups += [_msg("user", f"q{i}"), _msg("assistant", f"a{i}")]
    messages = [_msg("system", "SYSTEM")] + groups + [_msg("user", "CURRENT")]
    body = _body(messages)
    counts = [300000] * (len(groups) + 2) + [100000]
    with _mock_count_seq(counts):
        result = enforce_context_budget(body, {}, FAKE_MODEL)
    assert result["messages"][0]["role"] == "system"
    assert result["messages"][0]["content"] == "SYSTEM"


def test_last_user_message_never_dropped() -> None:
    messages = [_msg("system", "sys"), _msg("user", "old"), _msg("assistant", "old resp"), _msg("user", "MUST KEEP")]
    body = _body(messages)
    with _mock_count_seq([300000, 250000, 200000]):
        result = enforce_context_budget(body, {}, FAKE_MODEL)
    assert result["messages"][-1]["content"] == "MUST KEEP"


def test_atomic_tool_call_group_dropped_together() -> None:
    messages = [_msg("system", "sys"), *_tool_call_group("tc1", "result1"), _msg("user", "CURRENT")]
    body = _body(messages)
    with _mock_count_seq([300000, 300000, 200000]):
        result = enforce_context_budget(body, {}, FAKE_MODEL)
    roles = [m["role"] for m in result["messages"]]
    assert "tool" not in roles
    assert "system" in roles
    assert result["messages"][-1]["content"] == "CURRENT"


def test_stack_trace_tail_preserved_in_compressed_tool_result() -> None:
    noisy_prefix = "Running tests...\n" * 400
    error_at_end = "FAILED tests/test_foo.py::test_bar - AssertionError: assert 0 == 42"
    compressed = compress_tool_result_text(noisy_prefix + error_at_end)
    assert error_at_end in compressed


def test_final_translated_request_under_budget_after_compression() -> None:
    big_tool = "X" * (2 * _TOOL_COMPRESS_THRESHOLD_CHARS)
    messages = [_msg("system", "sys"), *_tool_call_group("tc1", big_tool), _msg("user", "CURRENT")]
    body = _body(messages)
    route = {"backend_max_context_tokens": 10000, "reserved_output_tokens": 1000, "tokenizer_safety_margin": 500}
    with _mock_count_seq([9000, 7000]):
        result = enforce_context_budget(body, route, FAKE_MODEL)
    tool_msg = next(m for m in result["messages"] if m.get("role") == "tool")
    assert _TRUNCATION_MARKER in tool_msg["content"]


def test_truncation_marker_visible_in_compressed_result() -> None:
    big = "line\n" * 5000
    assert "[truncated" in compress_tool_result_text(big).lower()


def test_output_budget_cap_does_not_zero_out_safe_value() -> None:
    oai_body = {"max_tokens": 5000}
    budget = ContextBudget(backend_max_context_tokens=10000, reserved_output_tokens=1000, tokenizer_safety_margin=500)
    result = _apply_output_budget_cap(oai_body, 20000, budget)
    assert result["max_tokens"] == 1024
