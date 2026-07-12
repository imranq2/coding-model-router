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
  claude_model_pattern  Regex checked on an exact claude_model lookup miss, so a
                        route keeps matching future Claude Code model-id version
                        bumps (e.g. "^claude-sonnet(-|$)") without a config edit.
  tokenizer_model       HuggingFace tokenizer id (e.g. "Qwen/Qwen3-Coder-30B-A3B-Instruct").
                        When set on an api_type: "openai" route, enables Strategy A —
                        exact token counting + preflight compression (context_manager.py)
                        instead of the character-heuristic (Strategy B).
  backend_max_context_tokens / reserved_output_tokens / tokenizer_safety_margin
                        Strategy A budget inputs — see context_manager.py docstring.

MODULE LAYOUT
-------------
This file is the FastAPI app + request orchestration only. Supporting logic lives
in sibling modules: constants.py, tool_sanitizer.py, aws_auth.py, bedrock_client.py,
route_config.py, message_translator.py, tokenizer.py, context_manager.py,
usage_stats.py, stream_converter.py.

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
import inspect
import json
import logging
import os
import re
import sys
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from aws_auth import _SigV4Auth, _sign_bedrock
from bedrock_client import _is_throttling_status, _send_with_bedrock_retry, _throttle_backoff_seconds
from constants import (
    _ANTHROPIC_ONLY_HEADERS,
    _CONTEXT_OVERFLOW_RE,
    _HOP_BY_HOP,
    _MAX_THROTTLE_RETRIES,
    _SKIP_HEADERS,
    _TOKEN_ESTIMATE_SAFETY_BUFFER,
)
from context_manager import enforce_context_budget
from message_translator import (
    _anthropic_to_openai_request,
    _estimate_input_tokens,
    _openai_to_anthropic_response,
)
from route_config import CONFIG_PATH, ROUTES, find_route
from stream_converter import _msg_id, _oai_stream_with_cleanup, _sse_event, _stream_logging
from tokenizer import count_oai_request_tokens
from tool_sanitizer import _sanitize_tools
from usage_stats import _record_tokens

# Logs go to stderr — start-model-router.sh redirects stderr to the log file.
# stdout is reserved for the live status line only.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)
log = logging.getLogger("model-router")


def _error_as_assistant_message(text: str, model: str, is_streaming: bool):
    """Return the error text as a valid Anthropic assistant message (status 200).

    Claude Code only shows text to the user when it receives a valid 200 response.
    HTTP 4xx/5xx are surfaced as opaque API errors with no actionable message.
    """
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


def _annotate_fallback_error(error_body: bytes, model: str) -> bytes:
    """Prefix an upstream error with a note explaining that this request had no
    configured route and went directly to Anthropic — without cost-routing or
    context-budget enforcement. Without this, a fallback request that later hits
    a real context-length error just looks like a bare, unexplained "context
    exceeded" failure with no link back to the actual root cause (a missing/stale
    route entry in models.json).
    """
    note = (
        f"[model-router] model '{model}' has no configured route — this request "
        "went directly to Anthropic without cost-routing or context-budget "
        "enforcement. Original error: "
    )
    try:
        parsed = json.loads(error_body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return note.encode() + error_body
    if not isinstance(parsed, dict):
        return note.encode() + error_body
    error_obj = parsed.get("error")
    if isinstance(error_obj, dict) and isinstance(error_obj.get("message"), str):
        error_obj["message"] = note + error_obj["message"]
        return json.dumps(parsed).encode()
    return note.encode() + error_body


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
      2. Body mutations  — rewrite model name, enforce context budget (Strategy A:
                           tokenizer-based preflight compression, or Strategy B:
                           character-heuristic dynamic cap), sanitize tool names,
                           inject chat_template_kwargs (local routes).
      3. Header building — auth strategy determines upstream Authorization header.
      4. Dispatch        — OpenAI routes use the openai SDK + Anthropic translation;
                           Anthropic/local routes stream bytes verbatim via httpx.
    """
    raw_body = await request.body()

    try:
        body_json = json.loads(raw_body)
    except json.JSONDecodeError:
        return JSONResponse({"error": "invalid JSON body"}, status_code=400)

    model = body_json.get("model", "")
    route = find_route(model)
    is_fallback_route = route is None
    request_id = _msg_id()

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
    # Passthrough routes forward the client's exact model id — Anthropic is
    # authoritative on which model ids exist, so a version-bumped id must never
    # be silently overridden by a possibly-stale pinned config value. Bedrock/local
    # routes DO rewrite to the configured backend model id, since that's a
    # genuinely different upstream model, not just a version.
    upstream_model = model if auth == "passthrough" else route["model"]
    is_streaming = bool(body_json.get("stream"))
    backend_label = {"none": "LOCAL", "aws": "BEDROCK"}.get(auth, "PASSTHROUGH")
    price_per_mtok: float = float(route.get("price_per_mtok", 0))
    tier: str = route.get("tier", "")
    tokenizer_model: str | None = route.get("tokenizer_model")

    # Neither OpenAI endpoints nor vllm-mlx (Qwen3VL) handle /count_tokens reliably;
    # return an estimate (exact, if a tokenizer is configured) so Claude Code doesn't
    # block on a 500.
    if req_suffix == "/count_tokens" and (api_type == "openai" or auth == "none"):
        if api_type == "openai" and tokenizer_model:
            oai_body_for_count = _anthropic_to_openai_request(body_json)
            token_count = count_oai_request_tokens(oai_body_for_count, tokenizer_model)
            if token_count is not None:
                return JSONResponse({"input_tokens": token_count})
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
    tool_name_map: dict[str, str] = {}

    # ── Context enforcement ───────────────────────────────────────────────────
    #
    # Two strategies, mutually exclusive:
    #
    # A. Tokenizer-based (preferred): translate to OpenAI format first, count tokens
    #    with the real HF tokenizer + chat template, then compress/drop messages if
    #    over budget. Activated when the route has a `tokenizer_model` field
    #    (Bedrock/Qwen routes only).
    #
    # B. Character-based (fallback): 4-chars-per-token heuristic with a 1.20x safety
    #    multiplier. Used for local routes, Anthropic-format Bedrock/passthrough
    #    routes, and any OpenAI-format route without a `tokenizer_model` configured.
    #
    if tokenizer_model and api_type == "openai":
        body_json = _anthropic_to_openai_request(body_json)
        body_json = enforce_context_budget(body_json, route, tokenizer_model)
        raw_body = json.dumps(body_json).encode()
    else:
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
                    smart_max = max(1024, int(raw_remaining * 0.70) - _TOKEN_ESTIMATE_SAFETY_BUFFER)
                    if route_max_tokens is not None:
                        smart_max = min(smart_max, route_max_tokens)
                    if requested_max > smart_max:
                        body_json["max_tokens"] = smart_max
                        body_changed = True
                        log.warning(
                            "[model-router] overflow: seeding max_tokens=%d "
                            "(raw_remaining=%d × 70%% − buffer)",
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
                effective_max_tokens = max(1, effective_max_tokens - _TOKEN_ESTIMATE_SAFETY_BUFFER)
                if requested_max > effective_max_tokens:
                    body_json["max_tokens"] = effective_max_tokens
                    body_changed = True
                    log.info(
                        "[model-router] dynamic max_tokens: capped %d → %d",
                        requested_max, effective_max_tokens,
                    )

        # Tool name sanitization is vllm-mlx specific (local routes only).
        if auth == "none":
            body_json, tool_name_map = _sanitize_tools(body_json)
            if tool_name_map:
                body_changed = True
                log.info("[model-router] sanitized %d tool name(s) for vllm-mlx", len(tool_name_map))

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
        base_headers = {k: v for k, v in base_headers.items() if k.lower() not in _ANTHROPIC_ONLY_HEADERS}

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
                # detect context-overflow 400s (Bedrock sends HTTP 200 + error event in
                # stream). This is a last-resort safety net: Strategy A's preflight
                # compression already avoids most overflows for tokenizer-configured
                # routes, but Strategy B routes (and any tokenizer under-estimate) still
                # need it. Bedrock's error reports "input = context_window + 1 - max_tokens"
                # (a formula, not the true count), so we can't compute the exact fix —
                # halve max_tokens on each attempt until it succeeds or retries exhaust.
                _MAX_OVERFLOW_RETRIES = 4
                first_chunk = None
                _original_max = oai_kwargs.get("max_tokens", route.get("context_window") or 32768)
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
                            # Handle both sync close() and async aclose()/close() — don't assume the
                            # openai SDK's stream.close() is always a coroutine.
                            _close_result = stream.close()
                            if inspect.isawaitable(_close_result):
                                await _close_result
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
            client, target_url, upstream_headers, raw_body, route, auth, request_id
        )
    except Exception:
        await client.aclose()
        raise

    # ── Log upstream errors so the root cause is visible in the router log ───
    if upstream_resp.status_code >= 400:
        error_body = await upstream_resp.aread()
        await client.aclose()
        log.error(
            "[model-router] request_id=%s upstream %d from %s: %s",
            request_id,
            upstream_resp.status_code,
            target_url,
            error_body[:1000].decode("utf-8", errors="replace"),
        )
        if is_fallback_route:
            error_body = _annotate_fallback_error(error_body, model)
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
