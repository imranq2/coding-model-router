"""Retry/backoff for Bedrock throttling.

Bedrock's on-demand ("Mantle") endpoints scale capacity gradually rather than
instantly — a request rate that ramps up faster than the endpoint can scale
(roughly, more than doubling within a ~30 minute window) gets rejected with a
throttling error even though the account is nowhere near its steady-state
quota. This is transient: retrying with backoff almost always succeeds once
Bedrock's autoscaler catches up.
"""
from __future__ import annotations

import asyncio
import logging
import random

import httpx

from aws_auth import _sign_bedrock
from constants import (
    _BEDROCK_MIN_DISPATCH_INTERVAL_S,
    _CONTEXT_OVERFLOW_RE,
    _MAX_THROTTLE_RETRIES,
    _THROTTLE_BASE_DELAY_S,
    _THROTTLE_MAX_DELAY_S,
    _THROTTLE_TEXT_RE,
)

log = logging.getLogger("model-router")

_bedrock_dispatch_lock = asyncio.Lock()
_bedrock_last_dispatch: float = 0.0  # asyncio monotonic time of last dispatch


async def _pace_bedrock_dispatch() -> None:
    global _bedrock_last_dispatch
    async with _bedrock_dispatch_lock:
        loop = asyncio.get_running_loop()
        wait = _BEDROCK_MIN_DISPATCH_INTERVAL_S - (loop.time() - _bedrock_last_dispatch)
        if wait > 0:
            await asyncio.sleep(wait)
        _bedrock_last_dispatch = asyncio.get_running_loop().time()


def _throttle_backoff_seconds(attempt: int) -> float:
    """Exponential backoff with full jitter (attempt is 0-indexed)."""
    ceiling = min(_THROTTLE_MAX_DELAY_S, _THROTTLE_BASE_DELAY_S * (2**attempt))
    return random.uniform(ceiling / 2, ceiling)


def _is_throttling_status(status_code: int, body_text: str = "") -> bool:
    """True only for transient Bedrock/AWS throttling responses worth blind-retrying.

    A 429 always indicates rate limiting and is always retried, even if its body
    happens to mention "input tokens" — a throttled request's error text is not
    evidence of a deterministic context-overflow failure. Context-window overflow
    is only excluded for non-429 4xx statuses, where it IS deterministic (retrying
    without reducing max_tokens will always fail again).
    """
    if status_code == 429:
        return True
    if status_code >= 400 and _CONTEXT_OVERFLOW_RE.search(body_text or ""):
        return False  # not transient — caller must reduce max_tokens before retrying
    if status_code >= 400 and _THROTTLE_TEXT_RE.search(body_text or ""):
        return True
    return False


async def _send_with_bedrock_retry(
    client: httpx.AsyncClient,
    target_url: str,
    upstream_headers: dict,
    raw_body: bytes,
    route: dict,
    auth: str,
    request_id: str = "unknown",
) -> httpx.Response:
    """POST to target_url, retrying on Bedrock throttling with backoff + re-signed headers.

    Only retries when auth == "aws" (Bedrock's on-demand endpoints are the ones that
    throttle when request rate ramps too fast — see module note above). Other backends
    (local, Anthropic passthrough) are sent once, unchanged, exactly as before.
    """
    attempt = 0
    while True:
        if auth == "aws":
            await _pace_bedrock_dispatch()
        upstream_req = client.build_request(
            "POST", target_url, headers=upstream_headers, content=raw_body
        )
        resp = await client.send(upstream_req, stream=True)

        if auth != "aws" or resp.status_code < 400 or attempt >= _MAX_THROTTLE_RETRIES:
            return resp

        error_body = await resp.aread()
        await resp.aclose()
        error_text = error_body.decode("utf-8", errors="replace")

        if not _is_throttling_status(resp.status_code, error_text):
            # Not a throttling error — hand back an in-memory Response carrying the
            # already-drained body so the caller's existing error-handling path
            # (which also calls .aread()) works unchanged.
            return httpx.Response(
                status_code=resp.status_code, headers=resp.headers, content=error_body
            )

        delay = _throttle_backoff_seconds(attempt)
        attempt += 1
        log.warning(
            "[model-router] request_id=%s Bedrock throttled (attempt %d/%d): backing off %.1fs — %s",
            request_id, attempt, _MAX_THROTTLE_RETRIES, delay, error_text[:200],
        )
        await asyncio.sleep(delay)
        # SigV4 signatures are time-scoped (~15 min skew tolerance) — re-sign before
        # retrying rather than reusing a signature computed before the backoff sleep.
        sig_headers = {k.lower(): v for k, v in _sign_bedrock(target_url, raw_body, route).items()}
        upstream_headers = {**upstream_headers, **sig_headers}
