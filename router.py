#!/usr/bin/env python3
"""
model-router/router.py — Anthropic Messages API proxy for Claude Code.

PURPOSE
-------
Claude Code sends every request to a single Anthropic Messages API endpoint.
This proxy intercepts those requests and re-routes them to cheaper/faster
backends (local GPU model, AWS Bedrock, or Anthropic direct) based on the
model name in the request body. This allows mixing local open-weight models,
cloud models, and direct Anthropic calls transparently.

ROUTING MODEL
-------------
Routes are keyed by `claude_model` (the name Claude Code sends). When a
request arrives, the router:
  1. Looks up the route by model name.
  2. Rewrites the model field to the upstream model name.
  3. Applies any body mutations (max_tokens, tool name sanitization,
     chat_template_kwargs injection).
  4. Forwards to the upstream using the route's auth strategy.
  5. Streams the response back verbatim (Anthropic routes) or translates
     it on the fly (OpenAI routes).

AUTH STRATEGIES (route.auth)
-----------------------------
  none        → Local vllm-mlx server; no auth needed.
  passthrough → Anthropic API direct; client Authorization header forwarded as-is.
  aws         → AWS Bedrock Mantle; SigV4-signed per request.

WIRE PROTOCOLS (route.api_type, default "anthropic")
-----------------------------------------------------
  anthropic   → Upstream speaks Anthropic Messages API. Bytes forwarded verbatim.
  openai      → Upstream speaks OpenAI Chat Completions API. The router translates
                the request from Anthropic format, then converts the response
                (including streaming SSE chunks) back to Anthropic format.

ROUTE CONFIG SCHEMA (~/model-router/router_config.json)
--------------------------------------------------------
Required fields:
  claude_model          Model name Claude Code sends (e.g. "claude-haiku-4-5-20251001")
  url                   Full upstream endpoint URL
  model                 Upstream model name (replaces claude_model in request body)
  auth                  Auth strategy: "none" | "passthrough" | "aws"

Optional fields:
  api_type              Wire protocol: "anthropic" (default) | "openai"
  aws_region            AWS region for Bedrock routes (default "us-east-1")
  tier                  Display tier: "haiku" | "sonnet" | "opus" | "fable"
                        Used for cost tracking labels only.
  price_per_mtok        Cost per million tokens for savings reporting (default 0)
  max_tokens            For local routes: ceiling (prevent KV-cache OOM).
                        For remote routes: floor (override Claude Code's 8K cap).
  chat_template_kwargs  Extra kwargs passed to the tokenizer's apply_chat_template.
                        For Qwen3 local routes: {"enable_thinking": false} disables
                        the reasoning chain, preventing slow "Thinking Process:" output.

EXAMPLE CONFIG
--------------
  { "routes": [
      { "tier": "haiku",
        "claude_model": "claude-haiku-4-5-20251001",
        "url": "http://localhost:8770/v1/messages",
        "model": "mlx-community/Qwen3.5-9B-MLX-4bit",
        "auth": "none",
        "price_per_mtok": 0,
        "chat_template_kwargs": {"enable_thinking": false} },
      { "tier": "sonnet",
        "claude_model": "claude-sonnet-4-6",
        "url": "https://bedrock-mantle.us-east-1.api.aws/v1/chat/completions",
        "model": "qwen.qwen3-coder-30b-a3b-v1:0",
        "auth": "aws", "aws_region": "us-east-1", "api_type": "openai",
        "price_per_mtok": 0.5 },
      { "tier": "opus",
        "claude_model": "claude-opus-4-8",
        "url": "https://api.anthropic.com/v1/messages",
        "model": "claude-opus-4-8",
        "auth": "passthrough",
        "price_per_mtok": 5.0 }
  ] }

THINKING SUPPRESSION
--------------------
Qwen3 reasoning models emit thinking chains by default. There are two separate
suppression mechanisms, one for each code path:

  Local routes (auth="none", vllm-mlx):
    The router injects chat_template_kwargs: {"enable_thinking": false} from the
    route config. vllm-mlx passes this to the tokenizer's Jinja2 chat template,
    which omits the <think> scaffolding — no thinking tokens are ever generated.

  Bedrock OpenAI routes (api_type="openai"):
    The _ThinkingStripper class strips <think>…</think> blocks from the streamed
    output. The model has already generated thinking tokens, but they are removed
    before being forwarded to Claude Code. This approach is used because Bedrock
    does not expose a pre-generation enable_thinking parameter.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import AsyncGenerator

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

from aws_auth import _SigV4Auth, _sign_bedrock
from bedrock_client import _is_throttling_status, _send_with_bedrock_retry, _throttle_backoff_seconds
from constants import (
    _ANTHROPIC_ONLY_HEADERS,
    _BEDROCK_MIN_DISPATCH_INTERVAL_S,
    _CONTEXT_OVERFLOW_RE,
    _HOP_BY_HOP,
    _MAX_THROTTLE_RETRIES,
    _OAI_TO_ANT_STOP,
    _SKIP_HEADERS,
    _THROTTLE_BASE_DELAY_S,
    _THROTTLE_MAX_DELAY_S,
    _THROTTLE_TEXT_RE,
)
from message_translator import (
    _anthropic_to_openai_request,
    _estimate_input_tokens,
    _openai_to_anthropic_response,
)
from tool_sanitizer import _sanitize_tools
from route_config import CONFIG, CONFIG_PATH, ROUTES, find_route

# Logs go to stderr — start-model-router.sh redirects stderr to the log file.
# stdout is reserved for the live status line only.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)
log = logging.getLogger("model-router")

_STDOUT_IS_TTY = sys.stdout.isatty()

# Reference price for savings comparison — read from the opus route in config, fallback to $5/MTok.
_OPUS_PRICE_PER_MTOK: float = next(
    (float(r.get("price_per_mtok", 5.0)) for r in CONFIG.get("routes", []) if r.get("tier") == "opus"),
    5.0,
)


# ---------------------------------------------------------------------------
# OpenAI <-> Anthropic translation (for api_type: "openai" routes)
# ---------------------------------------------------------------------------


class _ThinkingStripper:
    """Strip <think>…</think> blocks from streamed output on Bedrock OpenAI routes.

    Used only for api_type="openai" routes where the model has already generated
    thinking tokens — see "THINKING SUPPRESSION" in the module docstring for why
    this path exists alongside the chat_template_kwargs approach for local routes.

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
        """Return any buffered visible content at stream end.

        Content held back because it might be the start of a <think> tag is
        safe to forward once the stream is done (a partial tag cannot be completed).
        If the stream ended mid-think the buffered thinking content is discarded.
        """
        if self._inside:
            self._buf = ""
            self._inside = False
            return ""
        result, self._buf = self._buf, ""
        return result

    def _safe_forward_len(self) -> int:
        """Number of leading bytes in self._buf that are safe to forward now.

        Holds back the shortest suffix that could still be a prefix of _OPEN so
        we never emit a partial '<think' that will turn out to be a tag boundary.
        """
        tag = self._OPEN
        for i in range(1, len(tag)):
            if self._buf.endswith(tag[:i]):
                return len(self._buf) - i
        return len(self._buf)


def _msg_id() -> str:
    return "msg_" + os.urandom(12).hex()


# Per-model cumulative token counters {upstream_model: {"input": int, "output": int, "price_per_mtok": float}}
_token_stats: dict[str, dict] = {}

# Maps tier names to short display labels used in the status line (e.g. "haiku" → "low").
_TIER_LABEL = {"haiku": "low", "sonnet": "med", "opus": "high", "fable": "top"}
_TIER_ORDER = ["low", "med", "high", "top"]


def _record_tokens(upstream_model: str, in_tok: int, out_tok: int, price_per_mtok: float, backend_label: str, tier: str = "") -> None:
    """Update cumulative stats and emit a compact status line.

    Detailed per-model breakdown goes to the log file (stderr).
    A one-line tier summary overwrites the current terminal line on stdout —
    this is the "savings ticker" visible while Claude Code is running.
    """
    log.info(
        "[model-router] tokens  in=%-6d out=%-6d backend=%-12s model=%s",
        in_tok, out_tok, backend_label, upstream_model,
    )
    entry = _token_stats.setdefault(upstream_model, {"input": 0, "output": 0, "price_per_mtok": price_per_mtok, "tier": tier})
    entry["input"] += in_tok
    entry["output"] += out_tok
    entry["price_per_mtok"] = price_per_mtok

    # Detailed per-model totals → log file only
    grand_total = sum(s["input"] + s["output"] for s in _token_stats.values())
    grand_cost = sum((s["input"] + s["output"]) / 1_000_000 * s["price_per_mtok"] for s in _token_stats.values())
    grand_opus_cost = grand_total / 1_000_000 * _OPUS_PRICE_PER_MTOK
    log.info("[model-router] ── running totals ──────────────────────────────────────────────────")
    for mdl, s in _token_stats.items():
        mtok = s["input"] + s["output"]
        pct = 100.0 * mtok / grand_total if grand_total else 0.0
        cost = mtok / 1_000_000 * s["price_per_mtok"]
        cost_str = "FREE    " if s["price_per_mtok"] == 0 else f"${cost:.4f}"
        saved = mtok / 1_000_000 * (_OPUS_PRICE_PER_MTOK - s["price_per_mtok"])
        saved_str = f"  saved ${saved:.4f}" if saved > 0 else ""
        tier_label = _TIER_LABEL.get(s.get("tier", ""), s.get("tier", ""))
        log.info("[model-router]   %-4s %-46s %8d tok  %5.1f%%  %s%s", tier_label, mdl[:46], mtok, pct, cost_str, saved_str)
    total_saved = grand_opus_cost - grand_cost
    log.info(
        "[model-router]   %-4s %-46s %8d tok  100.0%%  $%.4f total  (saved $%.4f)",
        "", "ALL MODELS", grand_total, grand_cost, total_saved,
    )

    # Compact tier summary → stdout, single updating line
    by_tier: dict[str, dict] = {}
    for s in _token_stats.values():
        label = _TIER_LABEL.get(s.get("tier", ""), s.get("tier", "") or "?")
        e = by_tier.setdefault(label, {"tokens": 0, "cost": 0.0, "saved": 0.0})
        mtok2 = s["input"] + s["output"]
        e["tokens"] += mtok2
        e["cost"] += mtok2 / 1_000_000 * s["price_per_mtok"]
        savings = mtok2 / 1_000_000 * (_OPUS_PRICE_PER_MTOK - s["price_per_mtok"])
        if savings > 0:
            e["saved"] += savings

    parts = []
    for label in _TIER_ORDER:
        if label not in by_tier:
            continue
        e = by_tier[label]
        cost_str = "FREE" if e["cost"] == 0 else f"${e['cost']:.4f}"
        parts.append(f"{label}: {e['tokens']:,} tok {cost_str}")
    total_saved2 = sum(e["saved"] for e in by_tier.values())
    if total_saved2 > 0:
        parts.append(f"saved: ${total_saved2:.4f}")
    line = "  |  ".join(parts)
    if _STDOUT_IS_TTY:
        print(f"\033[2K\r{line}", end="", flush=True)
    else:
        print(line, flush=True)


def _sse_event(event_type: str, data: dict) -> bytes:
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n".encode()


def _error_as_assistant_message(text: str, model: str, is_streaming: bool):
    """Return the error text as a valid Anthropic assistant message (status 200).

    Claude Code only shows text to the user when it receives a valid 200 response.
    HTTP 4xx/5xx are surfaced as opaque API errors with no actionable message.
    """
    from fastapi.responses import JSONResponse, StreamingResponse

    msg_id = _msg_id()
    if not is_streaming:
        return JSONResponse({
            "id": msg_id, "type": "message", "role": "assistant",
            "content": [{"type": "text", "text": text}],
            "model": model, "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": len(text.split())},
        })

    async def _stream():
        yield _sse_event("message_start", {
            "type": "message_start",
            "message": {
                "id": msg_id, "type": "message", "role": "assistant",
                "content": [], "model": model,
                "stop_reason": None, "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        })
        yield _sse_event("ping", {"type": "ping"})
        yield _sse_event("content_block_start", {
            "type": "content_block_start", "index": 0,
            "content_block": {"type": "text", "text": ""},
        })
        yield _sse_event("content_block_delta", {
            "type": "content_block_delta", "index": 0,
            "delta": {"type": "text_delta", "text": text},
        })
        yield _sse_event("content_block_stop", {"type": "content_block_stop", "index": 0})
        yield _sse_event("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": len(text.split())},
        })
        yield _sse_event("message_stop", {"type": "message_stop"})

    return StreamingResponse(_stream(), status_code=200, media_type="text/event-stream")


async def _stream_oai_sdk_to_anthropic(
    stream,  # openai.AsyncStream[ChatCompletionChunk]
    msg_id: str,
    upstream_model: str,
    backend_label: str = "BEDROCK",
    price_per_mtok: float = 0.0,
    tier: str = "",
    first_chunk=None,  # pre-fetched chunk from peek-before-commit retry logic
) -> AsyncGenerator[bytes, None]:
    """Convert an openai SDK async stream to Anthropic SSE format.

    Uses typed ChatCompletionChunk objects instead of raw SSE text, so there
    is no manual JSON parsing — field access is via the SDK's dataclass attrs.
    """
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
                        "id": msg_id,
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                        "model": upstream_model,
                        "stop_reason": None,
                        "stop_sequence": None,
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
                                "type": "content_block_start",
                                "index": text_idx,
                                "content_block": {"type": "text", "text": ""},
                            })
                        yield _sse_event("content_block_delta", {
                            "type": "content_block_delta",
                            "index": text_idx,
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
                            "type": "content_block_start",
                            "index": ant_idx,
                            "content_block": {
                                "type": "tool_use",
                                "id": tc_id,
                                "name": tc_name,
                                "input": {},
                            },
                        })
                    ant_idx = tool_idx_map[oai_tc_idx]
                    if tc.function and tc.function.arguments:
                        yield _sse_event("content_block_delta", {
                            "type": "content_block_delta",
                            "index": ant_idx,
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
        await stream.close()

    # Always emit proper SSE termination events — both on clean completion and after errors.
    # If the stream yielded zero chunks (empty response or early error), synthesize message_start
    # so what follows is a valid Anthropic SSE sequence.
    if not sent_message_start:
        yield _sse_event("message_start", {
            "type": "message_start",
            "message": {
                "id": msg_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": upstream_model,
                "stop_reason": None,
                "stop_sequence": None,
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
                "type": "content_block_start",
                "index": text_idx,
                "content_block": {"type": "text", "text": ""},
            })
        yield _sse_event("content_block_delta", {
            "type": "content_block_delta",
            "index": text_idx,
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


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="model-router", docs_url=None, redoc_url=None)


def _passthrough_headers(request: Request) -> dict:
    """All client headers except ones we rebuild ourselves."""
    return {k: v for k, v in request.headers.items() if k.lower() not in _SKIP_HEADERS}


@app.get("/v1/models")
async def list_models():
    """Synthetic model list so Claude Code's startup probe doesn't 404."""
    return {"object": "list", "data": [{"id": m, "object": "model"} for m in ROUTES]}


@app.get("/health")
async def health():
    return {"status": "ok", "routes": len(ROUTES)}


@app.post("/v1/messages")
@app.post("/v1/messages/count_tokens")
async def proxy_messages(request: Request) -> StreamingResponse:
    """Handle all Claude Code API requests and route them to the appropriate backend.

    Flow:
      1. Route lookup    — find config for the requested model (fallback: Anthropic direct).
      2. Body mutations  — rewrite model name, enforce max_tokens, sanitize tool names,
                           inject chat_template_kwargs (local routes).
      3. Header building — auth strategy determines upstream Authorization header.
      4. Dispatch        — OpenAI routes use the openai SDK + Anthropic translation;
                           Anthropic/local routes stream bytes verbatim via httpx.
    """
    raw_body = await request.body()

    try:
        body_json = json.loads(raw_body)
    except json.JSONDecodeError:
        from fastapi.responses import JSONResponse

        return JSONResponse({"error": "invalid JSON body"}, status_code=400)

    model = body_json.get("model", "")
    route = find_route(model)

    # Suffix is "" for /v1/messages or "/count_tokens" for the count_tokens endpoint.
    req_suffix = request.url.path[len("/v1/messages"):]

    if route is None:
        log.warning("[model-router] unknown model '%s' — forwarding to Anthropic direct", model)
        route = {
            "auth": "passthrough",
            "url": f"https://api.anthropic.com/v1/messages{req_suffix}",
            "model": model,
        }

    auth = route["auth"]
    api_type = route.get("api_type", "anthropic")
    upstream_model = route["model"]
    is_streaming = bool(body_json.get("stream"))
    backend_label = {"none": "LOCAL", "aws": "BEDROCK"}.get(auth, "PASSTHROUGH")
    price_per_mtok: float = float(route.get("price_per_mtok", 0))
    tier: str = route.get("tier", "")

    # Neither OpenAI endpoints nor vllm-mlx (Qwen3VL) handle /count_tokens reliably;
    # return a rough estimate so Claude Code doesn't block on a 500.
    if req_suffix == "/count_tokens" and (
        api_type == "openai" or auth == "none"
    ):
        from fastapi.responses import JSONResponse

        return JSONResponse({"input_tokens": len(json.dumps(body_json)) // 4})

    # For Anthropic routes the URL has the suffix appended (supports /count_tokens).
    # For OpenAI routes the URL already points to /v1/chat/completions.
    if api_type == "openai":
        target_url = route["url"]
    else:
        target_url = route["url"] + req_suffix

    # ── Body mutations ────────────────────────────────────────────────────────
    body_changed = False
    if upstream_model != model:
        body_json["model"] = upstream_model
        body_changed = True

    # route_max_tokens is the upstream model's hard output ceiling — the most it can generate
    # in one call (e.g. 32768 for Qwen3-30B-A3B, model-specific for Bedrock). Used as a
    # ceiling: never let max_tokens in the forwarded request exceed what the upstream accepts.
    # When absent from the route config, leave the request unchanged.
    route_max_tokens: int | None = route.get("max_tokens")

    # Dynamic context window enforcement — calculate remaining tokens based on input size.
    # This prevents context overflow by capping output tokens when the input is large.
    route_context_window: int | None = route.get("context_window")
    if route_context_window is not None and auth != "none":
        # Estimate input tokens. The char-based heuristic (4 chars ≈ 1 token) undercounts
        # because it misses structural tokens added by the model's chat template: role
        # delimiters, tool-call markers, BOS/EOS tokens, and JSON punctuation that tokenizes
        # at higher density than prose. Apply a 20% safety multiplier so the cap fires before
        # Bedrock rejects the request for exceeding the context window.
        estimated_input_tokens = int(_estimate_input_tokens(body_json) * 1.20)
        remaining_tokens = route_context_window - estimated_input_tokens
        requested_max = body_json.get("max_tokens", 0)
        log.info(
            "[model-router] context window: estimated_input=%d remaining=%d requested=%d",
            estimated_input_tokens, remaining_tokens, requested_max,
        )
        if remaining_tokens <= 0:
            # The 1.20x safety multiplier pushed the estimate over the context window.
            # Recover the raw (pre-multiplier) estimate and use 70% of its remaining
            # capacity as max_tokens — much closer to the real working value than
            # the route ceiling (65536), so the Bedrock retry loop needs fewer halvings.
            raw_estimate = int(estimated_input_tokens / 1.20)
            raw_remaining = route_context_window - raw_estimate
            log.warning(
                "[model-router] 1.20x estimate (%d) >= context_window (%d); "
                "raw_estimate=%d raw_remaining=%d",
                estimated_input_tokens, route_context_window, raw_estimate, raw_remaining,
            )
            if raw_remaining > 0:
                smart_max = max(1024, int(raw_remaining * 0.70))
                if route_max_tokens is not None:
                    smart_max = min(smart_max, route_max_tokens)
                if requested_max > smart_max:
                    body_json["max_tokens"] = smart_max
                    body_changed = True
                    log.warning(
                        "[model-router] overflow: seeding max_tokens=%d "
                        "(raw_remaining=%d × 70%%)",
                        smart_max, raw_remaining,
                    )
            # If raw_remaining <= 0 the input is genuinely too large for any output;
            # let the route_max_tokens ceiling apply and the Bedrock response will be
            # an irrecoverable overflow error surfaced directly to Claude Code.
        else:
            if route_max_tokens is not None:
                effective_max_tokens = min(route_max_tokens, remaining_tokens)
            else:
                effective_max_tokens = remaining_tokens
            if requested_max > effective_max_tokens:
                body_json["max_tokens"] = effective_max_tokens
                body_changed = True
                log.info(
                    "[model-router] dynamic max_tokens: capped %d → %d",
                    requested_max, effective_max_tokens,
                )

    tool_name_map: dict[str, str] = {}

    # Tool name sanitization is vllm-mlx specific (local routes only).
    if auth == "none":
        body_json, tool_name_map = _sanitize_tools(body_json)
        if tool_name_map:
            body_changed = True
            log.info(
                "[model-router] sanitized %d tool name(s) for vllm-mlx", len(tool_name_map)
            )

    # route_max_tokens is the model's hard output limit — enforce as a ceiling for all
    # route types. Never raise max_tokens here: for remote routes the dynamic context
    # window cap above may have already reduced it below route_max_tokens to fit within
    # the remaining context, and raising it back would undo that work.
    if route_max_tokens is not None:
        if body_json.get("max_tokens", 0) > route_max_tokens:
            body_json["max_tokens"] = route_max_tokens
            body_changed = True

    # Inject chat_template_kwargs from route config (e.g. {"enable_thinking": false} for Qwen3
    # to disable the reasoning chain that would otherwise appear as visible output).
    route_chat_template_kwargs: dict | None = route.get("chat_template_kwargs")
    if route_chat_template_kwargs and api_type != "openai":
        merged = {**route_chat_template_kwargs, **body_json.get("chat_template_kwargs", {})}
        body_json["chat_template_kwargs"] = merged
        body_changed = True

    if api_type == "openai":
        # Translate Anthropic → OpenAI; re-encode immediately.
        body_json = _anthropic_to_openai_request(body_json)
        raw_body = json.dumps(body_json).encode()
    elif body_changed:
        raw_body = json.dumps(body_json).encode()

    # ── Build upstream headers ────────────────────────────────────────────────
    base_headers = _passthrough_headers(request)
    base_headers["content-type"] = "application/json"

    # Strip Anthropic-specific headers before forwarding to OpenAI-format endpoints.
    if api_type == "openai":
        base_headers = {
            k: v for k, v in base_headers.items() if k.lower() not in _ANTHROPIC_ONLY_HEADERS
        }

    if auth == "none":
        upstream_headers = base_headers
        log.info("[model-router] %s -> LOCAL      url=%s  model=%s", model, target_url, upstream_model)

    elif auth == "aws":
        region = route.get("aws_region", "us-east-1")
        if api_type == "openai":
            # OpenAI-format Bedrock routes sign per-request via _SigV4Auth on the httpx client.
            # Skip the eager _sign_bedrock call to avoid a NoCredentialsError crash (e.g. on
            # expired SSO) before the openai.APIStatusError handler is even reached.
            upstream_headers = base_headers
        else:
            # Anthropic-format Bedrock routes: pre-sign the headers here.
            # Normalize to lowercase to avoid duplicate content-type.
            try:
                sig_headers = {k.lower(): v for k, v in _sign_bedrock(target_url, raw_body, route).items()}
            except Exception as _cred_exc:
                from botocore.exceptions import TokenRetrievalError
                if isinstance(_cred_exc, TokenRetrievalError):
                    _profile = os.environ.get("AWS_PROFILE", "<profile>")
                    _msg = f"Your AWS Bedrock session has expired. Run: aws sso login --profile {_profile}"
                    return _error_as_assistant_message(_msg, upstream_model, is_streaming)
                raise
            upstream_headers = {**base_headers, **sig_headers}
            upstream_headers["content-type"] = "application/json"
            upstream_headers["anthropic-version"] = "2023-06-01"
        log.info(
            "[model-router] %s -> BEDROCK     region=%s  model=%s  api=%s",
            model, region, upstream_model, api_type,
        )

    else:  # passthrough — forward client Authorization verbatim
        upstream_headers = base_headers
        if auth_val := request.headers.get("authorization"):
            upstream_headers["authorization"] = auth_val
        log.info("[model-router] %s -> PASSTHROUGH url=%s  model=%s", model, target_url, upstream_model)

    # ── OpenAI route: use openai SDK (typed chunks, SigV4 via httpx.Auth) ─────
    if api_type == "openai":
        import openai
        from fastapi.responses import JSONResponse, Response

        # Strip /chat/completions suffix to get the base URL the SDK appends to.
        base_url = target_url.removesuffix("/chat/completions")
        _auth = _SigV4Auth(route) if auth == "aws" else None
        http_client = httpx.AsyncClient(auth=_auth, timeout=None)
        oai_client = openai.AsyncOpenAI(
            api_key="dummy",   # SigV4 / no-auth routes don't need a real key
            base_url=base_url,
            http_client=http_client,
            max_retries=0,     # disable SDK retries — our retry loop controls backoff
        )

        # body_json is already OpenAI-format (translated above); exclude `stream`
        # since we control it explicitly via the SDK kwarg.
        oai_kwargs = {k: v for k, v in body_json.items() if k != "stream"}
        msg_id = _msg_id()
        log.info("[model-router] openai request body: %s", json.dumps(oai_kwargs, default=str)[:2000])
        streaming_started = False
        try:
            if is_streaming:
                # Peek at the first chunk before committing to StreamingResponse so we can
                # detect context-overflow 400s (Bedrock sends HTTP 200 + error event in stream).
                # Bedrock's error reports "input = context_window + 1 - max_tokens" (a formula,
                # not the true count), so we can't compute the exact fix — instead we halve
                # max_tokens on each attempt until it succeeds or we exhaust retries.
                _MAX_OVERFLOW_RETRIES = 4
                first_chunk = None
                _original_max = oai_kwargs.get("max_tokens", route_context_window or 32768)
                _overflow_attempt = 0
                _throttle_attempt = 0
                while True:
                    if _overflow_attempt > 0 or _throttle_attempt > 0:
                        await http_client.aclose()
                        http_client = httpx.AsyncClient(auth=_auth, timeout=None)
                        oai_client = openai.AsyncOpenAI(api_key="dummy", base_url=base_url, http_client=http_client, max_retries=0)
                    stream = None
                    try:
                        # create() is inside the try so throttle errors it raises (after
                        # SDK-level retries) are caught by the same retry/backoff logic as
                        # errors from __anext__().
                        stream = await oai_client.chat.completions.create(**oai_kwargs, stream=True, stream_options={"include_usage": True})
                        first_chunk = await stream.__anext__()
                        break  # success — proceed with this stream
                    except Exception as peek_exc:
                        if stream is not None:
                            await stream.close()
                        _status = getattr(peek_exc, "status_code", None)
                        # Prefer raw response body for regex matching (more reliable than str(exc))
                        _resp = getattr(peek_exc, "response", None)
                        _peek_text = (getattr(_resp, "text", None) or str(peek_exc))
                        if _CONTEXT_OVERFLOW_RE.search(str(peek_exc)) and _overflow_attempt < _MAX_OVERFLOW_RETRIES:
                            # Halve max_tokens on each attempt: e.g. 26368 → 13184 → 6592 → 3296
                            # Bedrock's error encodes "N = context_window + 1 - max_tokens" (a
                            # formula, not the true input count), so we cannot reliably detect
                            # irrecoverable overflow from the error text — just halve and retry.
                            _overflow_attempt += 1
                            oai_kwargs["max_tokens"] = max(1, _original_max >> _overflow_attempt)
                            log.warning(
                                "[model-router] context overflow (attempt %d/%d): reducing max_tokens → %d",
                                _overflow_attempt, _MAX_OVERFLOW_RETRIES, oai_kwargs["max_tokens"],
                            )
                        elif (
                            _status is not None
                            and _is_throttling_status(_status, _peek_text)
                            and _throttle_attempt < _MAX_THROTTLE_RETRIES
                        ):
                            delay = _throttle_backoff_seconds(_throttle_attempt)
                            _throttle_attempt += 1
                            log.warning(
                                "[model-router] Bedrock throttled (attempt %d/%d): backing off %.1fs",
                                _throttle_attempt, _MAX_THROTTLE_RETRIES, delay,
                            )
                            await asyncio.sleep(delay)
                        else:
                            raise
                streaming_started = True
                return StreamingResponse(
                    _oai_stream_with_cleanup(stream, msg_id, upstream_model, backend_label, http_client, price_per_mtok, tier, first_chunk=first_chunk),
                    status_code=200,
                    media_type="text/event-stream",
                )
            else:
                _throttle_attempt = 0
                while True:
                    try:
                        resp = await oai_client.chat.completions.create(**oai_kwargs, stream=False)
                        break
                    except openai.APIStatusError as exc:
                        if (
                            _is_throttling_status(exc.status_code, exc.response.text)
                            and _throttle_attempt < _MAX_THROTTLE_RETRIES
                        ):
                            delay = _throttle_backoff_seconds(_throttle_attempt)
                            _throttle_attempt += 1
                            log.warning(
                                "[model-router] Bedrock throttled (attempt %d/%d): backing off %.1fs",
                                _throttle_attempt, _MAX_THROTTLE_RETRIES, delay,
                            )
                            await asyncio.sleep(delay)
                            continue
                        raise
                ant_resp = _openai_to_anthropic_response(resp.model_dump(), msg_id, upstream_model)
                usage = ant_resp.get("usage", {})
                _record_tokens(upstream_model, usage.get("input_tokens", 0), usage.get("output_tokens", 0), price_per_mtok, backend_label, tier)
                return JSONResponse(ant_resp)
        except openai.APIStatusError as exc:
            log.error(
                "[model-router] upstream %d from %s: %s",
                exc.status_code, target_url, exc.response.text,
            )
            # Surface the upstream error as a visible assistant message so the user
            # sees it in Claude Code instead of getting a silent empty response.
            try:
                err_body = exc.response.json()
                err_msg = (
                    err_body.get("error", {}).get("message")
                    or err_body.get("message")
                    or exc.message
                )
            except Exception:
                err_msg = exc.message or str(exc)
            return _error_as_assistant_message(
                f"Bedrock error ({exc.status_code}): {err_msg}",
                upstream_model, is_streaming,
            )
        except Exception as exc:
            from botocore.exceptions import TokenRetrievalError
            _cause = exc.__cause__ or exc.__context__
            if isinstance(exc, TokenRetrievalError) or isinstance(_cause, TokenRetrievalError):
                _profile = os.environ.get("AWS_PROFILE", "<profile>")
                _msg = f"Your AWS Bedrock session has expired. Run: aws sso login --profile {_profile}"
                return _error_as_assistant_message(_msg, upstream_model, is_streaming)
            raise
        finally:
            # Streaming: the generator owns cleanup after the first chunk flows.
            # Non-streaming and error paths: close here since we never yield a generator.
            if not streaming_started:
                await http_client.aclose()

    # ── Anthropic / local route: open upstream connection and stream verbatim ─
    # (For auth == "aws" this transparently retries Bedrock throttling with backoff —
    # see _send_with_bedrock_retry. Local/passthrough routes are sent once, unchanged.)
    client = httpx.AsyncClient(timeout=None)
    try:
        upstream_resp = await _send_with_bedrock_retry(
            client, target_url, upstream_headers, raw_body, route, auth
        )
    except Exception:
        await client.aclose()
        raise

    # ── Log upstream errors so the root cause is visible in the router log ───
    if upstream_resp.status_code >= 400:
        error_body = await upstream_resp.aread()
        await client.aclose()
        log.error(
            "[model-router] upstream %d from %s: %s",
            upstream_resp.status_code,
            target_url,
            error_body[:1000].decode("utf-8", errors="replace"),
        )
        from fastapi.responses import Response

        return Response(
            content=error_body,
            status_code=upstream_resp.status_code,
            media_type=upstream_resp.headers.get("content-type", "application/json"),
        )

    # ── Anthropic route: strip hop-by-hop headers and stream verbatim ─────────
    resp_headers = {
        k: v for k, v in upstream_resp.headers.items() if k.lower() not in _HOP_BY_HOP
    }

    return StreamingResponse(
        _stream_logging(upstream_resp, client, tool_name_map, backend_label, upstream_model, is_streaming, price_per_mtok, tier),
        status_code=upstream_resp.status_code,
        headers=resp_headers,
        media_type=upstream_resp.headers.get("content-type", "text/event-stream"),
    )


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


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("ROUTER_PORT", 8771))
    log.info("[model-router] starting on 127.0.0.1:%d  (config: %s)", port, CONFIG_PATH)
    for m, r in ROUTES.items():
        log.info("  %-40s -> %-12s  url=%s", m, r["auth"], r["url"])
    print(f"[model-router] listening on 127.0.0.1:{port}", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
