"""SSE streaming translation: OpenAI SDK stream -> Anthropic SSE, and passthrough streaming."""
from __future__ import annotations

import inspect
import json
import logging
import os
from typing import AsyncGenerator

import httpx

from constants import _OAI_TO_ANT_STOP
from usage_stats import _record_tokens

log = logging.getLogger("model-router")


class _ThinkingStripper:
    """Strip <think>…</think> blocks from streamed output on Bedrock OpenAI routes.

    Used only for api_type="openai" routes where the model has already generated
    thinking tokens — see "THINKING SUPPRESSION" in router.py's module docstring
    for why this path exists alongside the chat_template_kwargs approach for local
    routes.

    Processes text incrementally so tag boundaries that fall mid-chunk are handled
    correctly across multiple feed() calls.
    """

    _OPEN = "<think>"
    _CLOSE = "</think>"

    def __init__(self) -> None:
        self._buf = ""
        self._inside = False

    def feed(self, text: str) -> str:
        """Return the visible portion of *text* (thinking content is discarded)."""
        self._buf += text
        out: list[str] = []

        while True:
            if self._inside:
                end = self._buf.find(self._CLOSE)
                if end == -1:
                    self._buf = ""  # still inside thinking block — discard
                    break
                self._inside = False
                self._buf = self._buf[end + len(self._CLOSE):]
                if self._buf.startswith("\n"):  # drop the newline that typically follows </think>
                    self._buf = self._buf[1:]
            else:
                start = self._buf.find(self._OPEN)
                if start == -1:
                    # No open tag — safe to forward up to the point where a partial
                    # tag prefix could be hiding at the very end of the buffer.
                    safe = self._safe_forward_len()
                    out.append(self._buf[:safe])
                    self._buf = self._buf[safe:]
                    break
                out.append(self._buf[:start])
                self._inside = True
                self._buf = self._buf[start + len(self._OPEN):]

        return "".join(out)

    def flush(self) -> str:
        """Return any buffered visible content at stream end."""
        if self._inside:
            self._buf = ""
            self._inside = False
            return ""
        result, self._buf = self._buf, ""
        return result

    def _safe_forward_len(self) -> int:
        tag = self._OPEN
        for i in range(1, len(tag)):
            if self._buf.endswith(tag[:i]):
                return len(self._buf) - i
        return len(self._buf)


def _msg_id() -> str:
    return "msg_" + os.urandom(12).hex()


def _sse_event(event_type: str, data: dict) -> bytes:
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n".encode()


async def _stream_oai_sdk_to_anthropic(
    stream,  # openai.AsyncStream[ChatCompletionChunk]
    msg_id: str,
    upstream_model: str,
    backend_label: str = "BEDROCK",
    price_per_mtok: float = 0.0,
    tier: str = "",
    first_chunk=None,  # pre-fetched chunk from peek-before-commit retry logic
) -> AsyncGenerator[bytes, None]:
    """Convert an openai SDK async stream to Anthropic SSE format."""
    sent_message_start = False
    open_blocks: dict[int, dict] = {}
    next_idx = 0
    text_idx: int | None = None
    tool_idx_map: dict[int, int] = {}  # openai tool index → anthropic block index
    finish_reason: str | None = None
    input_tokens = 0
    output_tokens = 0
    thinking_stripper = _ThinkingStripper()

    async def _iter_stream():
        if first_chunk is not None:
            yield first_chunk
        async for chunk in stream:
            yield chunk

    try:
        async for chunk in _iter_stream():
            if not sent_message_start:
                sent_message_start = True
                yield _sse_event("message_start", {
                    "type": "message_start",
                    "message": {
                        "id": msg_id, "type": "message", "role": "assistant",
                        "content": [], "model": upstream_model,
                        "stop_reason": None, "stop_sequence": None,
                        "usage": {"input_tokens": 0, "output_tokens": 1},
                    },
                })
                yield _sse_event("ping", {"type": "ping"})

            for choice in chunk.choices:
                delta = choice.delta
                if choice.finish_reason:
                    finish_reason = choice.finish_reason

                if delta.content:
                    visible = thinking_stripper.feed(delta.content)
                    if visible:
                        if text_idx is None:
                            text_idx = next_idx
                            next_idx += 1
                            open_blocks[text_idx] = {"type": "text"}
                            yield _sse_event("content_block_start", {
                                "type": "content_block_start", "index": text_idx,
                                "content_block": {"type": "text", "text": ""},
                            })
                        yield _sse_event("content_block_delta", {
                            "type": "content_block_delta", "index": text_idx,
                            "delta": {"type": "text_delta", "text": visible},
                        })

                for tc in delta.tool_calls or []:
                    oai_tc_idx = tc.index
                    if oai_tc_idx not in tool_idx_map:
                        ant_idx = next_idx
                        next_idx += 1
                        tool_idx_map[oai_tc_idx] = ant_idx
                        tc_id = tc.id or f"toolu_{ant_idx:04x}"
                        tc_name = (tc.function.name if tc.function else "") or ""
                        open_blocks[ant_idx] = {"type": "tool_use"}
                        yield _sse_event("content_block_start", {
                            "type": "content_block_start", "index": ant_idx,
                            "content_block": {"type": "tool_use", "id": tc_id, "name": tc_name, "input": {}},
                        })
                    ant_idx = tool_idx_map[oai_tc_idx]
                    if tc.function and tc.function.arguments:
                        yield _sse_event("content_block_delta", {
                            "type": "content_block_delta", "index": ant_idx,
                            "delta": {"type": "input_json_delta", "partial_json": tc.function.arguments},
                        })

            if chunk.usage:
                if chunk.usage.prompt_tokens is not None:
                    input_tokens = chunk.usage.prompt_tokens
                if chunk.usage.completion_tokens is not None:
                    output_tokens = chunk.usage.completion_tokens

    except Exception as _exc:
        log.error("[model-router] upstream stream error: %s", _exc)
        _stream_error_msg = str(_exc)
    else:
        _stream_error_msg = None
    finally:
        # Handle both sync close() and async aclose()/close() — don't assume the
        # openai SDK's stream.close() is always a coroutine.
        close_method = getattr(stream, "aclose", getattr(stream, "close", None))
        if close_method is not None:
            _close_result = close_method()
            if inspect.isawaitable(_close_result):
                await _close_result

    # Always emit proper SSE termination events — both on clean completion and after errors.
    # If the stream yielded zero chunks (empty response or early error), synthesize message_start
    # so what follows is a valid Anthropic SSE sequence.
    if not sent_message_start:
        yield _sse_event("message_start", {
            "type": "message_start",
            "message": {
                "id": msg_id, "type": "message", "role": "assistant",
                "content": [], "model": upstream_model,
                "stop_reason": None, "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        })
        yield _sse_event("ping", {"type": "ping"})

    # When the stream errored before producing any content, emit a visible error text block
    # so the user knows what happened instead of receiving a silent empty response.
    if _stream_error_msg and not sent_message_start:
        yield _sse_event("content_block_start", {
            "type": "content_block_start", "index": 0,
            "content_block": {"type": "text", "text": ""},
        })
        yield _sse_event("content_block_delta", {
            "type": "content_block_delta", "index": 0,
            "delta": {"type": "text_delta", "text": f"[model-router error] {_stream_error_msg}"},
        })
        yield _sse_event("content_block_stop", {"type": "content_block_stop", "index": 0})

    # Flush any text the stripper held back waiting for a possible </think> continuation.
    remaining = thinking_stripper.flush()
    if remaining:
        if text_idx is None:
            text_idx = next_idx
            next_idx += 1
            open_blocks[text_idx] = {"type": "text"}
            yield _sse_event("content_block_start", {
                "type": "content_block_start", "index": text_idx,
                "content_block": {"type": "text", "text": ""},
            })
        yield _sse_event("content_block_delta", {
            "type": "content_block_delta", "index": text_idx,
            "delta": {"type": "text_delta", "text": remaining},
        })

    for idx in sorted(open_blocks):
        yield _sse_event("content_block_stop", {"type": "content_block_stop", "index": idx})

    stop_reason = _OAI_TO_ANT_STOP.get(finish_reason or "stop", "end_turn")
    yield _sse_event("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
        "usage": {"output_tokens": output_tokens},
    })
    yield _sse_event("message_stop", {"type": "message_stop"})
    _record_tokens(upstream_model, input_tokens, output_tokens, price_per_mtok, backend_label, tier)


async def _oai_stream_with_cleanup(
    stream,
    msg_id: str,
    upstream_model: str,
    backend_label: str,
    http_client: httpx.AsyncClient,
    price_per_mtok: float = 0.0,
    tier: str = "",
    first_chunk=None,
) -> AsyncGenerator[bytes, None]:
    """Wrap _stream_oai_sdk_to_anthropic and close the httpx client when the stream ends."""
    try:
        async for chunk in _stream_oai_sdk_to_anthropic(stream, msg_id, upstream_model, backend_label, price_per_mtok, tier, first_chunk=first_chunk):
            yield chunk
    finally:
        await http_client.aclose()


async def _stream_logging(
    resp: httpx.Response,
    client: httpx.AsyncClient,
    tool_name_map: dict[str, str],
    backend_label: str,
    upstream_model: str,
    is_streaming: bool,
    price_per_mtok: float = 0.0,
    tier: str = "",
) -> AsyncGenerator[bytes, None]:
    """Stream bytes, restore tool names, and log token usage at completion."""
    input_tokens = 0
    output_tokens = 0
    partial_sse = ""
    body_bytes = b""

    try:
        async for chunk in resp.aiter_bytes():
            text: str | None = None
            if tool_name_map:
                text = chunk.decode("utf-8", errors="replace")
                for sanitized, original in tool_name_map.items():
                    text = text.replace(f'"name":"{sanitized}"', f'"name":"{original}"')
                    text = text.replace(f'"name": "{sanitized}"', f'"name": "{original}"')
                chunk = text.encode("utf-8")

            if is_streaming:
                partial_sse += text if text is not None else chunk.decode("utf-8", errors="replace")
                while "\n\n" in partial_sse:
                    event_text, partial_sse = partial_sse.split("\n\n", 1)
                    for line in event_text.split("\n"):
                        if line.startswith("data: "):
                            try:
                                ev = json.loads(line[6:])
                                etype = ev.get("type")
                                if etype == "message_start":
                                    usage = ev.get("message", {}).get("usage", {})
                                    input_tokens = usage.get("input_tokens", input_tokens)
                                elif etype == "message_delta":
                                    usage = ev.get("usage", {})
                                    output_tokens = usage.get("output_tokens", output_tokens)
                            except (json.JSONDecodeError, KeyError):
                                pass
            else:
                body_bytes += chunk

            yield chunk
    finally:
        if not is_streaming:
            try:
                usage = json.loads(body_bytes).get("usage", {})
                input_tokens = usage.get("input_tokens", 0)
                output_tokens = usage.get("output_tokens", 0)
            except (json.JSONDecodeError, AttributeError):
                pass
        _record_tokens(upstream_model, input_tokens, output_tokens, price_per_mtok, backend_label, tier)
        await resp.aclose()
        await client.aclose()
