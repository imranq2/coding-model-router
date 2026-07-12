"""
Context budget management and structured compression for Qwen routes.

Pipeline:
  1. Translate Anthropic request → OpenAI format (done by caller)
  2. Count tokens with Qwen tokenizer + chat template
  3. If within budget → send as-is
  4. Phase 1: head+tail compress oversized tool results, recount
  5. Phase 2: drop oldest message groups (newest first, system+last-user never dropped)
  6. Recount after each drop; stop when budget satisfied
  7. Log all compression actions with token deltas

Budget formula (all values configurable in route JSON):
  effective_input_tokens = backend_max_context_tokens
                         - reserved_output_tokens
                         - tokenizer_safety_margin
"""
from __future__ import annotations

import dataclasses
import json
import logging
from typing import Any

from tokenizer import count_oai_request_tokens

log = logging.getLogger("model-router")

_DEFAULT_BACKEND_MAX_CONTEXT = 262144
_DEFAULT_RESERVED_OUTPUT = 16384
_DEFAULT_SAFETY_MARGIN = 6000

# Tool results longer than this get head+tail compressed before any message dropping.
_TOOL_COMPRESS_THRESHOLD_CHARS = 8000  # ~2 000 tokens at 4 chars/tok
_TOOL_HEAD_CHARS = 1500
_TOOL_TAIL_CHARS = 3000
_TRUNCATION_MARKER = "[truncated large tool result: kept beginning and end]"


@dataclasses.dataclass
class ContextBudget:
    backend_max_context_tokens: int = _DEFAULT_BACKEND_MAX_CONTEXT
    reserved_output_tokens: int = _DEFAULT_RESERVED_OUTPUT
    tokenizer_safety_margin: int = _DEFAULT_SAFETY_MARGIN

    @property
    def effective_input_tokens(self) -> int:
        return (
            self.backend_max_context_tokens
            - self.reserved_output_tokens
            - self.tokenizer_safety_margin
        )


def build_budget(route: dict[str, Any]) -> ContextBudget:
    """Construct a ContextBudget from route-config values, applying defaults."""
    budget = ContextBudget(
        backend_max_context_tokens=route.get("backend_max_context_tokens", _DEFAULT_BACKEND_MAX_CONTEXT),
        reserved_output_tokens=route.get("reserved_output_tokens", _DEFAULT_RESERVED_OUTPUT),
        tokenizer_safety_margin=route.get("tokenizer_safety_margin", _DEFAULT_SAFETY_MARGIN),
    )
    if (explicit := route.get("effective_input_tokens")) is not None:
        budget.backend_max_context_tokens = (
            explicit + budget.reserved_output_tokens + budget.tokenizer_safety_margin
        )
    return budget


def compress_tool_result_text(
    text: str,
    head_chars: int = _TOOL_HEAD_CHARS,
    tail_chars: int = _TOOL_TAIL_CHARS,
    marker: str = _TRUNCATION_MARKER,
) -> str:
    """
    Replace the middle of a long tool result with a truncation marker.

    The tail is kept larger than the head because errors, assertion failures,
    and compiler diagnostics appear at the end of command output.
    """
    if len(text) <= head_chars + len(marker) + tail_chars:
        return text
    return text[:head_chars] + f"\n{marker}\n" + text[-tail_chars:]


def _compress_tool_messages(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    """Compress oversized tool-role messages in place (head+tail). Returns (messages, log_lines)."""
    log_lines: list[str] = []
    result: list[dict[str, Any]] = []
    for i, msg in enumerate(messages):
        if msg.get("role") != "tool":
            result.append(msg)
            continue
        content = msg.get("content", "")
        if not isinstance(content, str) or len(content) <= _TOOL_COMPRESS_THRESHOLD_CHARS:
            result.append(msg)
            continue
        compressed = compress_tool_result_text(content)
        result.append({**msg, "content": compressed})
        log_lines.append(
            f"  tool[{i}] id={msg.get('tool_call_id', '?')!r}: {len(content):,} → {len(compressed):,} chars"
        )
    return result, log_lines


def _group_conversation(
    messages: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, list[list[dict[str, Any]]], dict[str, Any] | None]:
    """
    Partition messages into (system_msg, groups, last_user_msg).

    Groups are atomic units for drop purposes: a plain message is a group of 1;
    an assistant message with tool_calls plus its subsequent tool responses form
    a single group. system_msg and last_user_msg are never dropped.
    """
    system_msg: dict[str, Any] | None = None
    last_user_msg: dict[str, Any] | None = None
    middle: list[dict[str, Any]] = []

    for msg in messages:
        if msg.get("role") == "system":
            system_msg = msg
        else:
            middle.append(msg)

    if middle and middle[-1].get("role") == "user":
        last_user_msg = middle.pop()

    groups: list[list[dict[str, Any]]] = []
    i = 0
    while i < len(middle):
        msg = middle[i]
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            group: list[dict[str, Any]] = [msg]
            i += 1
            while i < len(middle) and middle[i].get("role") == "tool":
                group.append(middle[i])
                i += 1
            groups.append(group)
        else:
            groups.append([msg])
            i += 1

    return system_msg, groups, last_user_msg


def _reassemble(
    system_msg: dict[str, Any] | None,
    groups: list[list[dict[str, Any]]],
    last_user_msg: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    msgs: list[dict[str, Any]] = []
    if system_msg:
        msgs.append(system_msg)
    for group in groups:
        msgs.extend(group)
    if last_user_msg:
        msgs.append(last_user_msg)
    return msgs


def _char_estimate_tokens(oai_body: dict[str, Any]) -> int:
    # 3.5 chars/token is more conservative than the 4-char nominal; code-heavy content
    # often tokenizes more densely and the nominal heuristic understimates on observed
    # traffic, causing backend context-limit errors.
    return int(len(json.dumps(oai_body)) / 3.5)


def _apply_output_budget_cap(oai_body: dict[str, Any], token_count: int, budget: ContextBudget) -> dict[str, Any]:
    """
    Tighten max_tokens based on the final estimated input token count.

    The safety margin is applied a second time here (it was already consumed by the
    input compression budget) to protect against the estimate being lower than the
    actual token count.
    """
    current = oai_body.get("max_tokens")
    if current is None:
        return oai_body
    safe = max(1024, budget.backend_max_context_tokens - token_count - 2 * budget.tokenizer_safety_margin)
    if current > safe:
        log.info(
            "[model-router] output cap after compression: %d → %d (backend=%d - tokens=%d - 2×margin=%d)",
            current, safe, budget.backend_max_context_tokens, token_count, budget.tokenizer_safety_margin,
        )
        return {**oai_body, "max_tokens": safe}
    return oai_body


def enforce_context_budget(oai_body: dict[str, Any], route: dict[str, Any], tokenizer_model: str) -> dict[str, Any]:
    """
    Enforce the context budget on a translated OpenAI-format request.

    Counts tokens with the Qwen tokenizer (chat template applied). If the request
    exceeds effective_input_tokens, compresses tool results and/or drops oldest
    message groups until it fits. Returns the (possibly modified) body dict.

    Falls back to a character estimate when the tokenizer is unavailable so
    compression still runs. Never modifies the input dict.
    """
    budget = build_budget(route)

    current_max_tokens = oai_body.get("max_tokens")
    if current_max_tokens is not None and current_max_tokens > budget.reserved_output_tokens:
        log.info(
            "[model-router] capping max_tokens %d → %d (reserved_output_tokens)",
            current_max_tokens, budget.reserved_output_tokens,
        )
        oai_body = {**oai_body, "max_tokens": budget.reserved_output_tokens}

    _raw_count = count_oai_request_tokens(oai_body, tokenizer_model)
    if _raw_count is None:
        _using_char_estimate = True
        token_count = _char_estimate_tokens(oai_body)
        log.warning(
            "[model-router] tokenizer '%s' unavailable; using 4-chars/token heuristic: %d estimated tokens",
            tokenizer_model, token_count,
        )
    else:
        _using_char_estimate = False
        token_count = _raw_count

    def _recount(body: dict[str, Any]) -> int:
        if _using_char_estimate:
            return _char_estimate_tokens(body)
        result = count_oai_request_tokens(body, tokenizer_model)
        return result if result is not None else _char_estimate_tokens(body)

    log.info(
        "[model-router] preflight: %d input tokens / %d budget (backend=%d reserved=%d margin=%d%s)",
        token_count, budget.effective_input_tokens, budget.backend_max_context_tokens,
        budget.reserved_output_tokens, budget.tokenizer_safety_margin,
        " [char-estimate]" if _using_char_estimate else "",
    )

    if token_count <= budget.effective_input_tokens:
        return _apply_output_budget_cap(oai_body, token_count, budget)

    log.warning(
        "[model-router] request exceeds budget by %d tokens (%d > %d); compressing",
        token_count - budget.effective_input_tokens, token_count, budget.effective_input_tokens,
    )

    messages = list(oai_body.get("messages", []))

    messages, log_lines = _compress_tool_messages(messages)
    if log_lines:
        log.info("[model-router] tool result compression:\n%s", "\n".join(log_lines))
        updated = {**oai_body, "messages": messages}
        new_count = _recount(updated)
        log.info("[model-router] after tool compression: %d tokens (saved %d)", new_count, token_count - new_count)
        token_count = new_count
        if token_count <= budget.effective_input_tokens:
            return _apply_output_budget_cap({**oai_body, "messages": messages}, token_count, budget)

    system_msg, groups, last_user_msg = _group_conversation(messages)
    dropped = 0

    while groups and token_count > budget.effective_input_tokens:
        dropped_group = groups.pop(0)
        dropped += len(dropped_group)
        messages = _reassemble(system_msg, groups, last_user_msg)
        token_count = _recount({**oai_body, "messages": messages})
        if token_count <= budget.effective_input_tokens:
            break

    if dropped:
        log.warning(
            "[model-router] dropped %d oldest message(s) to fit context budget; "
            "remaining groups: %d  final token count: %d",
            dropped, len(groups), token_count,
        )

    if token_count > budget.effective_input_tokens:
        log.error(
            "[model-router] request still over budget after full compression "
            "(%d > %d tokens); only system prompt + latest user message remain — "
            "sending anyway, upstream may reject",
            token_count, budget.effective_input_tokens,
        )

    log.info(
        "[model-router] final: %d input tokens  reserved_output=%d  budget=%d",
        token_count, budget.reserved_output_tokens, budget.effective_input_tokens,
    )

    return _apply_output_budget_cap({**oai_body, "messages": messages}, token_count, budget)
