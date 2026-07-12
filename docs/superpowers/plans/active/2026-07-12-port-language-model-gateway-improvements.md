# Port language-model-gateway improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split `router.py` (1521 lines) into the same module layout as `language-model-gateway/language_model_gateway/gateway/routers/model_routing/`, port four correctness fixes and one hardening constant found there, and add two new capabilities (tokenizer-based context budgeting, regex route-pattern fallback) — while preserving every baseline capability the gateway version dropped (local/vllm-mlx routes, tool-name sanitization, `chat_template_kwargs` injection, the stdout "savings ticker", and the reactive context-overflow halving-retry safety net).

**Architecture:** Extract cohesive groups of baseline functions into flat top-level modules (no package/`src` layout — matches this repo's existing flat-file convention), in dependency order, updating `router.py`'s imports after each extraction so it stays runnable throughout. The final task rewrites `router.py`'s request-handling function to layer in the two approved new capabilities and the four bug fixes on top of the now-modular code. Config schema gains optional fields (`claude_model_pattern`, `tokenizer_model`, `backend_max_context_tokens`, `reserved_output_tokens`, `tokenizer_safety_margin`); `install-model-router.sh` is updated to emit them and to bundle the new files.

**Tech Stack:** Python 3.11+, FastAPI, httpx, openai SDK, boto3/botocore (Bedrock SigV4), `transformers` (new — HuggingFace tokenizer, Bedrock/Qwen routes only), pytest (new — dev-only, for the two new pure-logic modules and the throttle-detection fix).

## Global Constraints

- No behavior regression for existing baseline capabilities: local/vllm-mlx routes (`auth: "none"`), tool-name sanitization, `chat_template_kwargs` injection, the stdout "savings ticker" (`_record_tokens`), and the reactive context-overflow halving-retry loop must all keep working exactly as today.
- Do NOT port: MongoDB usage tracking (`usage_tracker.py`/`account_directory.py`), OIDC auth (`oidcauthlib`), or OpenTelemetry instrumentation — these require infrastructure (MongoDB, an internal b.well OIDC library, an OTel collector) this single-user local-proxy repo doesn't have. This was an explicit user decision, not an oversight.
- Do NOT port the gateway's `ENABLE_COST_SAVINGS_INFO` env-gated response-footer feature or its removal of the terminal savings ticker — baseline's ticker stays as the primary UX.
- Preserve the module-level logger name `"model-router"` and the `[model-router]` log-message prefix in every module (baseline convention; do not adopt the gateway's `[coding-model-router]` prefix).
- All new files are flat, at repo root, alongside `router.py` — no package directory, no `__init__.py`. This matches how `install-model-router.sh` bundles/downloads files today and how `start-model-router.sh` invokes `python3 router.py` (sibling-module imports resolve automatically because Python prepends the script's own directory to `sys.path`).
- Every extraction task must leave `router.py` in a state that starts successfully and serves `/health` and `/v1/models` correctly — verify after each task, not just at the end.

---

## File Structure

New files (repo root, alongside existing `router.py`, `models.json`, `install-model-router.sh`):

| File | Responsibility |
|---|---|
| `constants.py` | Shared regexes, header allowlists, retry/backoff tuning constants |
| `tool_sanitizer.py` | vllm-mlx tool-name sanitization (baseline-only; gateway dropped this along with local-route support) |
| `aws_auth.py` | SigV4 signing (`_sign_bedrock`, `_SigV4Auth`) |
| `bedrock_client.py` | Bedrock dispatch pacing, throttle detection (fixed), retry-with-backoff |
| `route_config.py` | Route loading + lookup, now with regex-pattern fallback matching |
| `message_translator.py` | Anthropic↔OpenAI request/response translation (pure move, no behavior change) |
| `tokenizer.py` | HuggingFace tokenizer loading + exact token counting (new capability) |
| `context_manager.py` | Preflight context-budget enforcement: tool-result compression + oldest-message dropping (new capability) |
| `usage_stats.py` | Cumulative token stats + the stdout "savings ticker" (baseline-only; gateway replaced this with MongoDB, which we're not porting) |
| `stream_converter.py` | SSE streaming translation (`_ThinkingStripper`, `_stream_oai_sdk_to_anthropic`, `_stream_logging`) |
| `router.py` | FastAPI app, route dispatch, body mutations — slimmed to orchestration only |
| `tests/conftest.py` | Adds repo root to `sys.path` so flat modules import in tests |
| `tests/test_bedrock_client.py` | Locks in the throttle-detection bug fix |
| `tests/test_route_config.py` | Regex-pattern route fallback |
| `tests/test_context_manager.py` | Context-budget compression pipeline |
| `.gitignore` | Excludes `__pycache__/` and the dev test venv |

Modified: `models.json`, `install-model-router.sh`, `CLAUDE.md`.

---

### Task 1: Extract `constants.py`, `tool_sanitizer.py`, `aws_auth.py` + test infra

**Files:**
- Create: `constants.py`
- Create: `tool_sanitizer.py`
- Create: `aws_auth.py`
- Create: `tests/conftest.py`
- Create: `.gitignore`
- Modify: `router.py:98-274` (imports + delete extracted code)

**Interfaces:**
- Produces: `constants._OAI_TO_ANT_STOP`, `constants._ANTHROPIC_ONLY_HEADERS`, `constants._SKIP_HEADERS`, `constants._HOP_BY_HOP`, `constants._BEDROCK_MIN_DISPATCH_INTERVAL_S`, `constants._MAX_THROTTLE_RETRIES`, `constants._THROTTLE_BASE_DELAY_S`, `constants._THROTTLE_MAX_DELAY_S`, `constants._THROTTLE_TEXT_RE`, `constants._CONTEXT_OVERFLOW_RE`, `constants._TOKEN_ESTIMATE_SAFETY_BUFFER` (new, see step 3). `tool_sanitizer._sanitize_tools(body_json: dict) -> tuple[dict, dict]`. `aws_auth._sign_bedrock(url: str, body: bytes, route: dict) -> dict`, `aws_auth._SigV4Auth` (httpx.Auth subclass, `__init__(self, route: dict)`).

- [ ] **Step 1: Create `constants.py`**

```python
"""Shared constants — retry/backoff tuning, regexes, header allowlists."""
from __future__ import annotations

import re

# finish_reason/stop_reason mappings (OpenAI -> Anthropic)
_OAI_TO_ANT_STOP = {"stop": "end_turn", "tool_calls": "tool_use", "length": "max_tokens"}

# Headers that are Anthropic-specific and must not be forwarded to OpenAI endpoints
_ANTHROPIC_ONLY_HEADERS = frozenset({"anthropic-version", "anthropic-beta", "x-api-key"})

_SKIP_HEADERS = frozenset({"host", "content-length", "transfer-encoding", "authorization"})

_HOP_BY_HOP = frozenset({"content-encoding", "transfer-encoding", "connection", "keep-alive"})

# Bedrock dispatch rate gate — prevents on-demand capacity 503s at the source.
# Bedrock's autoscaler rejects traffic that ramps faster than ~2x per 30 min,
# so bursts (e.g. parallel tool calls after an idle period) trigger 503s.
_BEDROCK_MIN_DISPATCH_INTERVAL_S = 0.3  # ≤ ~3 new Bedrock dispatches/sec

_MAX_THROTTLE_RETRIES = 5
_THROTTLE_BASE_DELAY_S = 1.0
_THROTTLE_MAX_DELAY_S = 20.0

_THROTTLE_TEXT_RE = re.compile(
    r"throttl|too many requests|rate.?limit|try again later"
    r"|increase.*traffic|traffic.*increase"
    r"|on.?demand.capacity|exceed.*capacity|double faster",
    re.IGNORECASE,
)

# Matches Bedrock's context-window overflow error. Defined at module level so
# _is_throttling_status can explicitly exclude it — context overflow is a
# deterministic failure (input too large) that requires modifying the request,
# not a transient server-side error that resolves on retry.
_CONTEXT_OVERFLOW_RE = re.compile(r"contains at least (\d+) input tokens", re.IGNORECASE)

# Safety margin subtracted from the computed remaining-token cap to absorb estimation
# imprecision (4 chars ≈ 1 token is a heuristic; dense content like JSON and tool
# schemas can tokenize at higher density, causing the estimate to fall short by a few
# tokens even after the 1.20x multiplier).
_TOKEN_ESTIMATE_SAFETY_BUFFER = 100
```

- [ ] **Step 2: Create `tool_sanitizer.py`**

```python
"""Tool-name sanitization for local (vllm-mlx) routes.

vllm-mlx enforces ^[A-Za-z0-9_-]{1,64}$ on tool names; Claude Code's tool names
(e.g. "mcp__playwright__browser_click") can violate this. Sanitize on the way in;
the caller restores original names in the response stream.
"""
from __future__ import annotations

import hashlib
import re

_VALID_TOOL_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _sanitize_tool_name(name: str) -> str:
    """Map an arbitrary tool name to one that satisfies ^[A-Za-z0-9_-]{1,64}$."""
    sanitized = re.sub(r"[^A-Za-z0-9_-]", "_", name)
    if len(sanitized) <= 64:
        return sanitized
    # Stable 64-char form: first 55 chars + '_' + 8-char SHA-256 prefix
    h = hashlib.sha256(name.encode()).hexdigest()[:8]
    return sanitized[:55] + "_" + h


def _sanitize_tools(body_json: dict) -> tuple[dict, dict]:
    """Sanitize tool names for vllm-mlx's [A-Za-z0-9_-]{1,64} constraint.

    Returns (modified_body_json, {sanitized_name: original_name}).
    Only entries that needed sanitization appear in the map.
    """
    tools = body_json.get("tools")
    if not tools:
        return body_json, {}

    mapping: dict[str, str] = {}
    new_tools: list = []
    used: set[str] = set()

    for tool in tools:
        original_name = tool.get("name", "")
        if _VALID_TOOL_NAME_RE.match(original_name):
            new_tools.append(tool)
            used.add(original_name)
        else:
            sanitized = _sanitize_tool_name(original_name)
            base, i = sanitized, 1
            while sanitized in used:
                suffix = f"_{i}"
                sanitized = base[: 64 - len(suffix)] + suffix
                i += 1
            used.add(sanitized)
            mapping[sanitized] = original_name
            new_tools.append({**tool, "name": sanitized})

    if not mapping:
        return body_json, {}
    return {**body_json, "tools": new_tools}, mapping
```

- [ ] **Step 3: Create `aws_auth.py`**

```python
"""AWS SigV4 signing for Bedrock Mantle."""
from __future__ import annotations

import os

import httpx


def _sign_bedrock(url: str, body: bytes, route: dict) -> dict:
    """Return a headers dict with AWS SigV4 Authorization for a Bedrock POST."""
    import boto3
    from botocore.auth import SigV4Auth
    from botocore.awsrequest import AWSRequest

    # Profile comes from AWS_PROFILE env var (set in the user's shell before starting the router).
    # Not stored in router_config.json so the config is shareable without exposing profile names.
    profile = os.environ.get("AWS_PROFILE")
    region = route.get("aws_region", "us-east-1")

    session = boto3.Session(profile_name=profile) if profile else boto3.Session()
    creds = session.get_credentials().get_frozen_credentials()
    req = AWSRequest(
        method="POST",
        url=url,
        data=body,
        headers={"Content-Type": "application/json"},
    )
    SigV4Auth(creds, "bedrock", region).add_auth(req)
    return dict(req.headers)


class _SigV4Auth(httpx.Auth):
    """Apply AWS SigV4 signing to every request the openai SDK makes.

    httpx calls auth_flow synchronously before each send. At that point the
    openai SDK has already serialised the body to bytes, so request.content is
    available for body-hash computation.
    """

    def __init__(self, route: dict) -> None:
        self._route = route

    def auth_flow(self, request: httpx.Request):
        signed = _sign_bedrock(str(request.url), request.content, self._route)
        for k, v in signed.items():
            request.headers[k.lower()] = v
        yield request
```

- [ ] **Step 4: Create `tests/conftest.py`**

```python
from __future__ import annotations

import sys
from pathlib import Path

# Flat top-level modules (constants.py, route_config.py, etc.) live at repo root,
# not in a package — add it to sys.path so `import constants` etc. resolve.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
```

- [ ] **Step 5: Create `.gitignore`**

```
__pycache__/
*.pyc
.venv-dev/
```

- [ ] **Step 6: Remove the extracted code from `router.py` and import it instead**

Read `router.py`, then apply this edit (removes lines 171-271 — the tool-sanitization and SigV4 sections — and updates the import block):

Old (`router.py:98-114`):
```python
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import random
import re
import sys
from pathlib import Path
from typing import AsyncGenerator

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
```

New:
```python
from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
import sys
from pathlib import Path
from typing import AsyncGenerator

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

from aws_auth import _SigV4Auth, _sign_bedrock
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
from tool_sanitizer import _sanitize_tools
```

Then delete the now-duplicated block `router.py:171-271` (from the `# Tool name sanitization for local (vllm-mlx) routes` comment through the end of the `_SigV4Auth` class) — everything between the `find_route` function and the `# Retry/backoff for Bedrock throttling` comment. Also delete the now-duplicated constants inside the "Retry/backoff" section that step 1 already moved: `_BEDROCK_MIN_DISPATCH_INTERVAL_S`, `_MAX_THROTTLE_RETRIES`, `_THROTTLE_BASE_DELAY_S`, `_THROTTLE_MAX_DELAY_S`, `_THROTTLE_TEXT_RE`, `_CONTEXT_OVERFLOW_RE`, and `_OAI_TO_ANT_STOP` (leave the retry *functions* — `_pace_bedrock_dispatch`, `_bedrock_dispatch_lock`, `_bedrock_last_dispatch`, `_throttle_backoff_seconds`, `_is_throttling_status`, `_send_with_bedrock_retry` — in place for now; Task 2 extracts those). Also delete `_ANTHROPIC_ONLY_HEADERS` (currently defined at `router.py:482`) and `_SKIP_HEADERS`/`_HOP_BY_HOP` (at `router.py:1014,1417`) since those now come from `constants.py`.

- [ ] **Step 7: Verify `router.py` still starts and serves correctly**

```bash
cd /Users/imranqureshi/git/coding-model-router
python3 -c "import ast; ast.parse(open('router.py').read())"  # syntax check
ROUTER_CONFIG=/tmp/empty_router_config.json python3 -c '
import json
json.dump({"routes": []}, open("/tmp/empty_router_config.json", "w"))
'
ROUTER_CONFIG=/tmp/empty_router_config.json ROUTER_PORT=18771 python3 router.py &
sleep 1
curl -sf http://127.0.0.1:18771/health && echo " OK"
kill %1
```
Expected: `{"status":"ok","routes":0} OK`

- [ ] **Step 8: Commit**

```bash
git add constants.py tool_sanitizer.py aws_auth.py tests/conftest.py .gitignore router.py
git commit -m "Extract constants, tool_sanitizer, aws_auth into their own modules"
```

---

### Task 2: Extract `bedrock_client.py` + fix 429/context-overflow throttle-detection bug

**Files:**
- Create: `bedrock_client.py`
- Create: `tests/test_bedrock_client.py`
- Modify: `router.py` (imports + delete extracted code)

**Interfaces:**
- Consumes: `constants._BEDROCK_MIN_DISPATCH_INTERVAL_S`, `constants._CONTEXT_OVERFLOW_RE`, `constants._MAX_THROTTLE_RETRIES`, `constants._THROTTLE_BASE_DELAY_S`, `constants._THROTTLE_MAX_DELAY_S`, `constants._THROTTLE_TEXT_RE`, `aws_auth._sign_bedrock`.
- Produces: `bedrock_client._is_throttling_status(status_code: int, body_text: str = "") -> bool`, `bedrock_client._throttle_backoff_seconds(attempt: int) -> float`, `bedrock_client._send_with_bedrock_retry(client, target_url, upstream_headers, raw_body, route, auth, request_id: str = "unknown") -> httpx.Response`.

- [ ] **Step 1: Create `bedrock_client.py`**

This is baseline's existing retry/backoff logic, with the throttle-detection bug fixed: a `429` status now retries even when the error body happens to mention "input tokens" (a rate-limit response can legitimately include that phrase in its message; the old code treated any such body as a non-transient context-overflow error and refused to retry it). A `request_id` parameter is threaded through for log correlation.

```python
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
```

- [ ] **Step 2: Write the failing test for the fix**

```python
# tests/test_bedrock_client.py
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
```

- [ ] **Step 3: Install pytest and run the test**

```bash
cd /Users/imranqureshi/git/coding-model-router
python3 -m venv .venv-dev
.venv-dev/bin/pip install -q pytest httpx
.venv-dev/bin/python -m pytest tests/test_bedrock_client.py -v
```
Expected: all 6 tests PASS (the module already contains the fix from Step 1, so this confirms correctness rather than demonstrating red→green — the fix and its test are introduced together here since Task 1 hadn't extracted `bedrock_client.py` yet for a pre-fix baseline to exist as a separate module).

- [ ] **Step 4: Remove the extracted code from `router.py` and import it instead**

Old (`router.py:274-396`, the "Retry/backoff for Bedrock throttling" section, from the section comment through the end of `_send_with_bedrock_retry`):
```python
# ---------------------------------------------------------------------------
# Retry/backoff for Bedrock throttling
# ---------------------------------------------------------------------------
#
# Bedrock's on-demand ("Mantle") endpoints scale capacity gradually rather than
# ... [full section as read in Task 1's baseline, through the end of _send_with_bedrock_retry]
```

New: delete the whole section, and update the import block at the top of `router.py` to add:
```python
from bedrock_client import _is_throttling_status, _send_with_bedrock_retry, _throttle_backoff_seconds
```

Also remove `import random` from `router.py`'s import list (no longer used directly — `_throttle_backoff_seconds` now lives in `bedrock_client.py`); keep `import re` (still used by `_CONTEXT_OVERFLOW_RE.search` at the peek-retry loop, which Task 8 keeps in `router.py`, and by other regex use sites still in the file).

- [ ] **Step 5: Verify**

```bash
cd /Users/imranqureshi/git/coding-model-router
python3 -c "import ast; ast.parse(open('router.py').read())"
ROUTER_CONFIG=/tmp/empty_router_config.json ROUTER_PORT=18771 python3 router.py &
sleep 1
curl -sf http://127.0.0.1:18771/health && echo " OK"
kill %1
```

- [ ] **Step 6: Commit**

```bash
git add bedrock_client.py tests/test_bedrock_client.py router.py .venv-dev
git reset .venv-dev  # dev venv is gitignored, not committed — undo any accidental staging
git commit -m "Extract bedrock_client.py; fix 429 misclassified as context-overflow"
```

---

### Task 3: Create `route_config.py` with regex-pattern route fallback

**Files:**
- Create: `route_config.py`
- Create: `tests/test_route_config.py`
- Modify: `router.py` (imports + delete extracted code)

**Interfaces:**
- Produces: `route_config.CONFIG_PATH: Path`, `route_config.CONFIG: dict`, `route_config.ROUTES: dict[str, dict]`, `route_config.PATTERNS: list[tuple[re.Pattern, dict]]`, `route_config.find_route(model: str) -> dict | None`, `route_config._build_routes(config: dict) -> tuple[dict, list]`, `route_config._reload_routes() -> dict`.

This is a new capability (approved): a route can carry an optional `claude_model_pattern` regex so it keeps matching future Claude Code model-id version bumps (e.g. `claude-sonnet-4-6` → `claude-sonnet-5`) without a `models.json` edit + reinstall. Exact match is tried first (fast path); regex patterns are only consulted on a miss.

- [ ] **Step 1: Create `route_config.py`**

```python
"""Route loading and lookup.

Routes are keyed by `claude_model` (the model name Claude Code sends) for O(1)
exact-match lookup. A route may additionally carry a `claude_model_pattern` regex,
checked only on an exact-match miss, so a single route keeps matching future
Claude Code model-id version bumps without a models.json edit + reinstall.
"""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

log = logging.getLogger("model-router")

CONFIG_PATH = Path(
    os.environ.get("ROUTER_CONFIG", Path.home() / "model-router" / "router_config.json")
)


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return json.load(f)


def _build_routes(config: dict) -> tuple[dict[str, dict], list[tuple[re.Pattern, dict]]]:
    """Build the exact-match route dict and the ordered pattern fallback list."""
    routes: dict[str, dict] = {}
    patterns: list[tuple[re.Pattern, dict]] = []
    for route in config.get("routes", []):
        key = route["claude_model"]
        if key in routes:
            log.warning("[model-router] duplicate route for model '%s' — later entry wins", key)
        routes[key] = route
        if pattern := route.get("claude_model_pattern"):
            patterns.append((re.compile(pattern), route))
    return routes, patterns


try:
    CONFIG: dict = load_config()
except FileNotFoundError:
    log.error(
        "[model-router] config not found at %s — run install-model-router first; starting with no routes",
        CONFIG_PATH,
    )
    CONFIG = {"routes": []}

ROUTES: dict[str, dict]
PATTERNS: list[tuple[re.Pattern, dict]]
ROUTES, PATTERNS = _build_routes(CONFIG)


def find_route(model: str) -> dict | None:
    """Exact match first (fast path), then the first matching claude_model_pattern."""
    if route := ROUTES.get(model):
        return route
    for pattern, route in PATTERNS:
        if pattern.search(model):
            return route
    return None


def _reload_routes() -> dict[str, dict]:
    """Reload routes from disk and return the updated exact-match routes dict."""
    global CONFIG, ROUTES, PATTERNS
    CONFIG = load_config()
    ROUTES, PATTERNS = _build_routes(CONFIG)
    return ROUTES
```

- [ ] **Step 2: Create `tests/test_route_config.py`**

```python
"""Tests for route_config.py — exact and pattern-based model routing."""
from __future__ import annotations

import re
from unittest.mock import patch

from route_config import _build_routes, find_route


def test_build_routes_exact_key() -> None:
    config = {"routes": [{"claude_model": "claude-opus-4-8", "model": "upstream-opus"}]}
    routes, patterns = _build_routes(config)
    assert routes["claude-opus-4-8"]["model"] == "upstream-opus"
    assert patterns == []


def test_build_routes_compiles_pattern() -> None:
    config = {
        "routes": [
            {
                "claude_model": "claude-sonnet-5",
                "claude_model_pattern": "^claude-sonnet(-|$)",
                "model": "upstream-sonnet",
            }
        ]
    }
    _routes, patterns = _build_routes(config)
    assert len(patterns) == 1
    compiled, route = patterns[0]
    assert route["model"] == "upstream-sonnet"
    assert compiled.search("claude-sonnet-6")
    assert not compiled.search("claude-opus-4-8")


def test_find_route_prefers_exact_match_over_pattern() -> None:
    fake_routes = {"claude-a": {"model": "exact"}}
    fake_patterns = [(re.compile("^claude-a"), {"model": "pattern"})]
    with patch("route_config.ROUTES", fake_routes), patch("route_config.PATTERNS", fake_patterns):
        route = find_route("claude-a")
    assert route is not None
    assert route["model"] == "exact"


def test_find_route_falls_back_to_pattern_when_no_exact_match() -> None:
    fake_patterns = [(re.compile(r"^claude-sonnet(-|$)"), {"model": "sonnet-backend"})]
    with patch("route_config.ROUTES", {}), patch("route_config.PATTERNS", fake_patterns):
        route = find_route("claude-sonnet-6")
    assert route is not None
    assert route["model"] == "sonnet-backend"


def test_find_route_no_match_returns_none() -> None:
    with patch("route_config.ROUTES", {}), patch("route_config.PATTERNS", []):
        assert find_route("totally-unknown-model") is None
```

- [ ] **Step 3: Run the tests**

```bash
cd /Users/imranqureshi/git/coding-model-router
.venv-dev/bin/python -m pytest tests/test_route_config.py -v
```
Expected: all 5 tests PASS.

- [ ] **Step 4: Remove the extracted code from `router.py` and import it instead**

Old (`router.py:127-162`, the "Config" section, from the section comment through `find_route`):
```python
# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CONFIG_PATH = Path(
    os.environ.get("ROUTER_CONFIG", Path.home() / "model-router" / "router_config.json")
)


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return json.load(f)


try:
    CONFIG: dict = load_config()
except FileNotFoundError:
    log.error(
        "[model-router] config not found at %s — run install-model-router first; starting with no routes",
        CONFIG_PATH,
    )
    CONFIG = {"routes": []}

# Key routes by claude_model (what Claude Code sends) for O(1) lookup.
# Warn on duplicates rather than silently dropping earlier entries.
ROUTES: dict[str, dict] = {}
for _r in CONFIG.get("routes", []):
    _key = _r["claude_model"]
    if _key in ROUTES:
        log.warning("[model-router] duplicate route for model '%s' — later entry wins", _key)
    ROUTES[_key] = _r


def find_route(model: str) -> dict | None:
    return ROUTES.get(model)
```

New: delete this block. Add to the import section:
```python
from route_config import CONFIG_PATH, ROUTES, find_route
```
Leave the `_OPUS_PRICE_PER_MTOK` computation (`router.py:164-168`, which reads `CONFIG`) in place for now — Task 6 moves it into `usage_stats.py` and will update this import to `from route_config import CONFIG` there.

- [ ] **Step 5: Verify**

```bash
cd /Users/imranqureshi/git/coding-model-router
python3 -c "import ast; ast.parse(open('router.py').read())"
ROUTER_CONFIG=/tmp/empty_router_config.json ROUTER_PORT=18771 python3 router.py &
sleep 1
curl -sf http://127.0.0.1:18771/health && echo " OK"
kill %1
```

- [ ] **Step 6: Commit**

```bash
git add route_config.py tests/test_route_config.py router.py
git commit -m "Extract route_config.py; add claude_model_pattern regex fallback matching"
```

---

### Task 4: Extract `message_translator.py` (pure move, no behavior change)

**Files:**
- Create: `message_translator.py`
- Modify: `router.py` (imports + delete extracted code)

**Interfaces:**
- Consumes: `constants._OAI_TO_ANT_STOP`.
- Produces: `message_translator._anthropic_content_to_text`, `_estimate_input_tokens`, `_anthropic_to_openai_request`, `_convert_user_content`, `_openai_to_anthropic_response` — all identical signatures to baseline.

- [ ] **Step 1: Create `message_translator.py`**

```python
"""Anthropic <-> OpenAI request/response translation (for api_type: "openai" routes)."""
from __future__ import annotations

import json
import re

from constants import _OAI_TO_ANT_STOP


def _anthropic_content_to_text(content) -> str:
    """Flatten Anthropic content (str or list of blocks) to a plain string."""
    if isinstance(content, str):
        return content
    return "\n".join(b.get("text", "") for b in content if b.get("type") == "text")


def _estimate_input_tokens(body_json: dict) -> int:
    """Estimate input token count from Anthropic request body using character-based heuristic.

    Uses the standard heuristic: ~4 characters ≈ 1 token for English text.
    Counts all content: system, messages (text + tool_use + tool_result), and tool definitions.
    """
    total_chars = 0

    if system := body_json.get("system"):
        total_chars += len(_anthropic_content_to_text(system))

    for msg in body_json.get("messages", []):
        content = msg.get("content", "")
        if isinstance(content, str):
            total_chars += len(content)
        elif isinstance(content, list):
            for block in content:
                btype = block.get("type")
                if btype == "text":
                    total_chars += len(block.get("text", ""))
                elif btype == "tool_use":
                    total_chars += len(block.get("name", ""))
                    inp = block.get("input")
                    if inp:
                        total_chars += len(json.dumps(inp))
                elif btype == "tool_result":
                    result_content = block.get("content", "")
                    if isinstance(result_content, str):
                        total_chars += len(result_content)
                    elif isinstance(result_content, list):
                        for rb in result_content:
                            if rb.get("type") == "text":
                                total_chars += len(rb.get("text", ""))

    for tool in body_json.get("tools", []):
        total_chars += len(tool.get("name", ""))
        total_chars += len(tool.get("description", ""))
        schema = tool.get("input_schema") or tool.get("parameters") or {}
        if schema:
            total_chars += len(json.dumps(schema))

    return (total_chars + 3) // 4


def _convert_user_content(blocks: list) -> str | list:
    """Convert a list of non-tool-result Anthropic content blocks to OpenAI user content."""
    text_only = all(b.get("type") == "text" for b in blocks)
    if text_only:
        return "\n".join(b.get("text", "") for b in blocks)
    result = []
    for block in blocks:
        btype = block.get("type")
        if btype == "text":
            result.append({"type": "text", "text": block.get("text", "")})
        elif btype == "image":
            src = block.get("source", {})
            if src.get("type") == "base64":
                url = f"data:{src.get('media_type', 'image/jpeg')};base64,{src.get('data', '')}"
            else:
                url = src.get("url", "")
            result.append({"type": "image_url", "image_url": {"url": url}})
    return result


def _anthropic_to_openai_request(body_json: dict) -> dict:
    """Translate an Anthropic Messages API request body to OpenAI Chat Completions format."""
    oai: dict = {"model": body_json["model"]}

    for field in ("stream", "temperature", "top_p", "max_tokens"):
        if field in body_json:
            oai[field] = body_json[field]

    messages: list = []

    if system := body_json.get("system"):
        messages.append({"role": "system", "content": _anthropic_content_to_text(system)})

    for msg in body_json.get("messages", []):
        role = msg["role"]
        content = msg["content"]

        if role == "assistant":
            if isinstance(content, list):
                text_parts: list[str] = []
                tool_calls: list[dict] = []
                for block in content:
                    btype = block.get("type")
                    if btype == "text":
                        text_parts.append(block.get("text", ""))
                    elif btype == "tool_use":
                        tool_calls.append({
                            "id": block.get("id", f"call_{len(tool_calls)}"),
                            "type": "function",
                            "function": {
                                "name": block.get("name", ""),
                                "arguments": json.dumps(block.get("input", {})),
                            },
                        })
                oai_msg: dict = {"role": "assistant"}
                if text_parts:
                    oai_msg["content"] = "\n".join(text_parts)
                if tool_calls:
                    oai_msg["tool_calls"] = tool_calls
                messages.append(oai_msg)
            else:
                messages.append({"role": "assistant", "content": content or ""})

        elif role == "user":
            if isinstance(content, list):
                pending: list = []
                for block in content:
                    if block.get("type") == "tool_result":
                        if pending:
                            messages.append({"role": "user", "content": _convert_user_content(pending)})
                            pending = []
                        result_content = block.get("content", "")
                        messages.append({
                            "role": "tool",
                            "tool_call_id": block.get("tool_use_id", ""),
                            "content": _anthropic_content_to_text(result_content) if isinstance(result_content, list) else str(result_content or ""),
                        })
                    else:
                        pending.append(block)
                if pending:
                    messages.append({"role": "user", "content": _convert_user_content(pending)})
            else:
                messages.append({"role": "user", "content": content or ""})

    oai["messages"] = messages

    if tools := body_json.get("tools"):
        oai["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": t.get("name", ""),
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", {}),
                },
            }
            for t in tools
        ]

    if tc := body_json.get("tool_choice"):
        tc_type = tc.get("type")
        if tc_type == "auto":
            oai["tool_choice"] = "auto"
        elif tc_type == "any":
            oai["tool_choice"] = "required"
        elif tc_type == "none":
            oai["tool_choice"] = "none"
        elif tc_type == "tool":
            oai["tool_choice"] = {"type": "function", "function": {"name": tc.get("name", "")}}

    return oai


def _openai_to_anthropic_response(resp_json: dict, msg_id: str, upstream_model: str) -> dict:
    """Translate a non-streaming OpenAI Chat Completions response to Anthropic format."""
    usage = resp_json.get("usage", {})
    content: list = []
    stop_reason = "end_turn"

    choices = resp_json.get("choices", [])
    if choices:
        choice = choices[0]
        message = choice.get("message", {})
        stop_reason = _OAI_TO_ANT_STOP.get(choice.get("finish_reason", "stop"), "end_turn")

        if text := message.get("content"):
            # Qwen3 reasoning models embed <think>…</think> in the content field.
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip("\n").strip()
            if text:
                content.append({"type": "text", "text": text})

        for tc in message.get("tool_calls") or []:
            fn = tc.get("function", {})
            try:
                input_data = json.loads(fn.get("arguments", "{}"))
            except json.JSONDecodeError:
                input_data = {}
            content.append({
                "type": "tool_use",
                "id": tc.get("id", f"toolu_{len(content):04x}"),
                "name": fn.get("name", ""),
                "input": input_data,
            })

    return {
        "id": msg_id,
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": upstream_model,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }
```

- [ ] **Step 2: Remove the extracted code from `router.py` and import it instead**

Delete `router.py`'s `_anthropic_content_to_text`, `_estimate_input_tokens`, `_anthropic_to_openai_request`, `_convert_user_content`, `_openai_to_anthropic_response` function definitions (the block from `# ---... OpenAI <-> Anthropic translation ...` down through the end of `_openai_to_anthropic_response`, i.e. everything between `_error_as_assistant_message` and `_stream_oai_sdk_to_anthropic`).

Add to the import block:
```python
from message_translator import (
    _anthropic_to_openai_request,
    _estimate_input_tokens,
    _openai_to_anthropic_response,
)
```
(`_anthropic_content_to_text` and `_convert_user_content` are only called from within `message_translator.py` itself now, so `router.py` does not need to import them.)

- [ ] **Step 3: Verify**

```bash
cd /Users/imranqureshi/git/coding-model-router
python3 -c "import ast; ast.parse(open('router.py').read())"
ROUTER_CONFIG=/tmp/empty_router_config.json ROUTER_PORT=18771 python3 router.py &
sleep 1
curl -sf http://127.0.0.1:18771/health && echo " OK"
kill %1
```

- [ ] **Step 4: Commit**

```bash
git add message_translator.py router.py
git commit -m "Extract message_translator.py (pure move, no behavior change)"
```

---

### Task 5: Add `tokenizer.py` + `context_manager.py` (tokenizer-based context budget pipeline)

**Files:**
- Create: `tokenizer.py`
- Create: `context_manager.py`
- Create: `tests/test_context_manager.py`

**Interfaces:**
- Produces: `tokenizer.count_oai_request_tokens(oai_body: dict, tokenizer_model: str) -> int | None`. `context_manager.ContextBudget` (dataclass: `backend_max_context_tokens`, `reserved_output_tokens`, `tokenizer_safety_margin`, property `effective_input_tokens`), `context_manager.build_budget(route: dict) -> ContextBudget`, `context_manager.compress_tool_result_text(text, head_chars=1500, tail_chars=3000, marker=...) -> str`, `context_manager.enforce_context_budget(oai_body: dict, route: dict, tokenizer_model: str) -> dict`.
- Not wired into `router.py` yet — Task 8 does that. This task only adds the modules and proves them correct in isolation.

This is new capability, approved by the user: for Bedrock/Qwen routes with a `tokenizer_model` configured, count tokens exactly (via the model's own HuggingFace tokenizer + chat template) instead of the 4-chars-per-token heuristic, and if the request is still over budget, compress oversized tool results (head+tail truncation) and then drop the oldest message groups — before ever sending the request upstream. This directly targets the context-overflow problem CLAUDE.md's "Context window and token management" section already documents as a pain point, replacing baseline's purely reactive halve-and-retry with a preflight strategy that avoids most overflows in the first place.

- [ ] **Step 1: Create `tokenizer.py`**

```python
"""
Qwen tokenizer integration for accurate preflight token counting.

Loads the HuggingFace tokenizer for the configured backend model and counts tokens
by applying the model's chat template — capturing all formatting overhead (role
delimiters, BOS/EOS, tool-call markers, generation prompt) that the character-based
heuristic misses.

Tokenizer objects are cached after the first load so repeated requests are cheap.
"""
from __future__ import annotations

import functools
import json
import logging
from typing import Any

log = logging.getLogger("model-router")

# Models that have permanently failed to load — skip future attempts.
_UNAVAILABLE: set[str] = set()


def _flatten_message_content(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Normalize OpenAI-format message content to plain strings for apply_chat_template.

    The Qwen Jinja2 chat template expects content to be a string, but OpenAI-format
    messages may carry content as a list of typed blocks:
      [{"type": "text", "text": "..."}, {"type": "image_url", ...}]
    This function extracts text from each block and joins them so the tokenizer
    can process the message without a 'Can only get item pairs from a mapping' error.
    Tool-call and tool-result messages are also normalised.
    """
    result = []
    for msg in messages:
        msg = dict(msg)

        content = msg.get("content")
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, dict):
                    btype = block.get("type", "")
                    if btype == "text":
                        parts.append(block.get("text", ""))
                    elif btype in ("tool_result", "tool_use"):
                        inner = block.get("content") or block.get("output", "")
                        if isinstance(inner, list):
                            parts.extend(b.get("text", "") for b in inner if isinstance(b, dict))
                        else:
                            parts.append(str(inner))
                    else:
                        parts.append(f"[{btype}]")
                else:
                    parts.append(str(block))
            msg["content"] = "\n".join(parts)

        tool_calls = msg.get("tool_calls")
        if isinstance(tool_calls, list):
            fixed: list[dict[str, Any]] = []
            for tc in tool_calls:
                tc = dict(tc)
                fn = tc.get("function")
                if isinstance(fn, dict):
                    fn = dict(fn)
                    args = fn.get("arguments", "{}")
                    if isinstance(args, str):
                        try:
                            fn["arguments"] = json.loads(args)
                        except (json.JSONDecodeError, ValueError):
                            fn["arguments"] = {}
                    tc["function"] = fn
                fixed.append(tc)
            msg["tool_calls"] = fixed

        result.append(msg)
    return result


@functools.lru_cache(maxsize=8)
def _load_tokenizer(model_id: str) -> Any:
    """Load and cache a HuggingFace tokenizer. Logs on first use; cached forever."""
    from transformers import AutoTokenizer

    log.info("[model-router] loading tokenizer '%s' (cached after first use)", model_id)
    return AutoTokenizer.from_pretrained(model_id)


def count_oai_request_tokens(oai_body: dict[str, Any], tokenizer_model: str) -> int | None:
    """
    Count tokens for an OpenAI-format request using the Qwen tokenizer.

    Applies the model's Jinja2 chat template so the count includes all overhead:
    system/user/assistant role tokens, tool-schema encoding, generation prompt, etc.

    Returns None if the tokenizer cannot be loaded (e.g. transformers not installed,
    model not cached). The caller should fall back gracefully in that case.
    """
    if tokenizer_model in _UNAVAILABLE:
        return None
    try:
        tok = _load_tokenizer(tokenizer_model)
    except ImportError:
        _UNAVAILABLE.add(tokenizer_model)
        log.warning(
            "[model-router] `transformers` not installed; tokenizer-based token "
            "counting unavailable for '%s'. Install it: pip install transformers",
            tokenizer_model,
        )
        return None
    except Exception as exc:
        _UNAVAILABLE.add(tokenizer_model)
        log.warning(
            "[model-router] failed to load tokenizer '%s': %s — "
            "falling back to character-based estimation",
            tokenizer_model, exc,
        )
        return None

    messages: list[dict[str, Any]] = _flatten_message_content(oai_body.get("messages", []))
    tools: list[dict[str, Any]] | None = oai_body.get("tools") or None

    try:
        text: str = tok.apply_chat_template(
            messages, tools=tools, tokenize=False, add_generation_prompt=True,
        )
    except Exception as exc:
        if tools is not None:
            log.warning(
                "[model-router] apply_chat_template failed with tools for '%s': %s "
                "— retrying without tools argument",
                tokenizer_model, exc,
            )
            try:
                text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception as exc2:
                log.warning(
                    "[model-router] apply_chat_template failed for '%s': %s "
                    "— falling back to character-based estimation for this request",
                    tokenizer_model, exc2,
                )
                return None
        else:
            log.warning(
                "[model-router] apply_chat_template failed for '%s': %s "
                "— falling back to character-based estimation for this request",
                tokenizer_model, exc,
            )
            return None

    return len(tok.encode(text))
```

- [ ] **Step 2: Create `context_manager.py`**

```python
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
```

- [ ] **Step 3: Create `tests/test_context_manager.py`**

```python
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
```

- [ ] **Step 4: Run the tests**

```bash
cd /Users/imranqureshi/git/coding-model-router
.venv-dev/bin/python -m pytest tests/test_context_manager.py -v
```
Expected: all 22 tests PASS. No network calls or `transformers` install needed — the tokenizer call is mocked throughout.

- [ ] **Step 5: Commit**

```bash
git add tokenizer.py context_manager.py tests/test_context_manager.py
git commit -m "Add tokenizer.py + context_manager.py: preflight context-budget compression"
```

---

### Task 6: Extract `usage_stats.py` (savings ticker — baseline-only, not in the gateway version)

**Files:**
- Create: `usage_stats.py`
- Modify: `router.py` (imports + delete extracted code)

**Interfaces:**
- Consumes: `route_config.CONFIG`.
- Produces: `usage_stats._record_tokens(upstream_model: str, in_tok: int, out_tok: int, price_per_mtok: float, backend_label: str, tier: str = "") -> None`.

The gateway version replaced this entirely with MongoDB writes (not being ported — see Global Constraints). This task moves it unchanged; it is the primary UX for this single-user CLI tool and must keep working exactly as today.

- [ ] **Step 1: Create `usage_stats.py`**

```python
"""Cumulative token stats and the stdout "savings ticker".

Detailed per-model breakdown goes to the log file (stderr). A one-line tier
summary overwrites the current terminal line on stdout while Claude Code runs.
"""
from __future__ import annotations

import logging
import sys

from route_config import CONFIG

log = logging.getLogger("model-router")

_STDOUT_IS_TTY = sys.stdout.isatty()

# Reference price for savings comparison — read from the opus route in config, fallback to $5/MTok.
_OPUS_PRICE_PER_MTOK: float = next(
    (float(r.get("price_per_mtok", 5.0)) for r in CONFIG.get("routes", []) if r.get("tier") == "opus"),
    5.0,
)

# Per-model cumulative token counters {upstream_model: {"input": int, "output": int, "price_per_mtok": float}}
_token_stats: dict[str, dict] = {}

# Maps tier names to short display labels used in the status line (e.g. "haiku" → "low").
_TIER_LABEL = {"haiku": "low", "sonnet": "med", "opus": "high", "fable": "top"}
_TIER_ORDER = ["low", "med", "high", "top"]


def _record_tokens(upstream_model: str, in_tok: int, out_tok: int, price_per_mtok: float, backend_label: str, tier: str = "") -> None:
    """Update cumulative stats and emit a compact status line."""
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
```

- [ ] **Step 2: Remove the extracted code from `router.py` and import it instead**

Delete `router.py`'s `_STDOUT_IS_TTY` assignment (now unused there — remove it and the `_OPUS_PRICE_PER_MTOK` computation that Task 3 left in place), the `_token_stats`, `_TIER_LABEL`, `_TIER_ORDER` module globals, and the entire `_record_tokens` function.

Add to the import block:
```python
from usage_stats import _record_tokens
```
Remove `from route_config import CONFIG_PATH, ROUTES, find_route` → keep as-is (still needed for `CONFIG_PATH`/`ROUTES`/`find_route`, but `CONFIG` itself is no longer needed directly in `router.py` since `usage_stats.py` now owns the opus-price computation).

- [ ] **Step 3: Verify**

```bash
cd /Users/imranqureshi/git/coding-model-router
python3 -c "import ast; ast.parse(open('router.py').read())"
ROUTER_CONFIG=/tmp/empty_router_config.json ROUTER_PORT=18771 python3 router.py &
sleep 1
curl -sf http://127.0.0.1:18771/health && echo " OK"
kill %1
```

- [ ] **Step 4: Commit**

```bash
git add usage_stats.py router.py
git commit -m "Extract usage_stats.py (savings ticker, baseline-only)"
```

---

### Task 7: Extract `stream_converter.py` + fix `stream.close()` awaitable-safety bug

**Files:**
- Create: `stream_converter.py`
- Modify: `router.py` (imports + delete extracted code)

**Interfaces:**
- Consumes: `constants._OAI_TO_ANT_STOP`, `usage_stats._record_tokens`.
- Produces: `stream_converter._ThinkingStripper`, `_msg_id() -> str`, `_sse_event(event_type: str, data: dict) -> bytes`, `_stream_oai_sdk_to_anthropic(stream, msg_id, upstream_model, backend_label="BEDROCK", price_per_mtok=0.0, tier="", first_chunk=None) -> AsyncGenerator[bytes, None]`, `_oai_stream_with_cleanup(stream, msg_id, upstream_model, backend_label, http_client, price_per_mtok=0.0, tier="", first_chunk=None) -> AsyncGenerator[bytes, None]`, `_stream_logging(resp, client, tool_name_map, backend_label, upstream_model, is_streaming, price_per_mtok=0.0, tier="") -> AsyncGenerator[bytes, None]`.

The gateway version fixed a real robustness gap here: baseline unconditionally does `await stream.close()`, assuming the openai SDK's stream `close()` is always a coroutine. If a future SDK version (or a mocked/alternate stream implementation) provides a synchronous `close()`, that `await` raises `TypeError: object NoneType can't be used in 'await' expression`. This port fixes that in `_stream_oai_sdk_to_anthropic` **and** preserves baseline's correct behavior of tracking `output_tokens` in a local variable and reporting the real count in the final `message_delta` SSE event — the gateway version accidentally introduced a regression here (a leftover unused `output_tokens` local always stayed `0` after their refactor split usage into a separate `usage_sink` dict but forgot to route it back into the terminal event). Do not reproduce that regression.

- [ ] **Step 1: Create `stream_converter.py`**

```python
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
```

- [ ] **Step 2: Remove the extracted code from `router.py` and import it instead**

Delete `router.py`'s `_ThinkingStripper` class, `_msg_id`, `_stream_oai_sdk_to_anthropic`, `_oai_stream_with_cleanup`, and `_stream_logging` function definitions.

Add to the import block:
```python
from stream_converter import _msg_id, _oai_stream_with_cleanup, _sse_event, _stream_logging
```
(`_ThinkingStripper` and `_stream_oai_sdk_to_anthropic` are only used inside `stream_converter.py` itself, so `router.py` doesn't need to import them directly — `_oai_stream_with_cleanup` is the entry point router.py calls.)

- [ ] **Step 3: Verify**

```bash
cd /Users/imranqureshi/git/coding-model-router
python3 -c "import ast; ast.parse(open('router.py').read())"
ROUTER_CONFIG=/tmp/empty_router_config.json ROUTER_PORT=18771 python3 router.py &
sleep 1
curl -sf http://127.0.0.1:18771/health && echo " OK"
kill %1
```

- [ ] **Step 4: Commit**

```bash
git add stream_converter.py router.py
git commit -m "Extract stream_converter.py; fix stream.close() awaitable-safety bug"
```

---

### Task 8: Rewrite `router.py` — wire in Strategy A/B context enforcement, passthrough model-id fix, fallback-error annotation

**Files:**
- Modify: `router.py` (module docstring, imports, `proxy_messages`, new `_annotate_fallback_error`)

**Interfaces:**
- Consumes: everything produced by Tasks 1–7 (`bedrock_client`, `constants`, `context_manager`, `message_translator`, `route_config`, `stream_converter`, `tokenizer`, `tool_sanitizer`, `usage_stats`, `aws_auth`).
- Produces: no new public interface — this is the final orchestration layer. `_error_as_assistant_message` and `_passthrough_headers` remain in `router.py` (tightly coupled to the FastAPI request/response objects, not reusable elsewhere).

This task applies the two remaining bug fixes:
- **Passthrough model-id preservation**: `upstream_model` is now `model` (the client's exact requested id) for `auth == "passthrough"` routes, instead of always using the pinned `route["model"]` value from config. A hardcoded `claude-opus-4-8` in `models.json` must never silently override a version-bumped id Anthropic itself sends — Anthropic is authoritative on which model ids exist. Bedrock/local routes still rewrite to the configured backend model id (that's a genuinely different upstream model, not a version).
- **Fallback-error annotation**: when a request had no configured route at all (fell through to the hardcoded Anthropic-direct fallback) and the upstream then errors, the error body now carries an explicit note that this request skipped cost-routing and context-budget enforcement — otherwise a confusing unexplained context-overflow error has no link back to the real root cause (a stale/missing route entry).

It also wires in Strategy A (tokenizer-based preflight budget) for any route with `tokenizer_model` set, while keeping Strategy B (character heuristic) — now with the `_TOKEN_ESTIMATE_SAFETY_BUFFER` hardening constant — for every other route, including local/vllm-mlx routes which are explicitly excluded from context-window enforcement here exactly as they are in baseline. The reactive halving-retry loop for context overflow stays in place as a last-resort safety net alongside the a2 `stream.close()` fix.

- [ ] **Step 1: Update the module docstring**

Old (`router.py:1-97`, keep lines 1-81 and 97-98 unchanged; only the "ROUTE CONFIG SCHEMA" optional-fields list and the file needs a short module-layout note):

Find this block within the docstring:
```
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
```

Replace with:
```
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
```

- [ ] **Step 2: Verify the accumulated import block is correct**

By this point, `router.py`'s imports (assembled incrementally by Tasks 1–7) should read:

```python
from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import AsyncGenerator

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
```

Read `router.py` and reconcile the actual accumulated import block against this list — add `inspect` to the stdlib imports (needed for the a2 fix in the peek-retry loop below; it should already be present if Task 7 added it, but the peek-retry loop in this task's Step 4 also needs it directly in `router.py`), and add any of the above names not yet imported. Remove `import random` if still present (unused after Task 2). Keep `import re` (used by `_CONTEXT_OVERFLOW_RE.search(str(peek_exc))` in the peek-retry loop below).

- [ ] **Step 3: Rewrite the "Body mutations" section of `proxy_messages` to add Strategy A/B branching, the a5 fix, and the a6 fix**

Read `router.py` to find the current `proxy_messages` function (built up incrementally by prior tasks — it should closely match baseline's structure at this point, since Tasks 1-7 only changed *where* helper functions live, not `proxy_messages` itself). Replace the whole function body with:

```python
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
```

- [ ] **Step 4: Add `_annotate_fallback_error`**

Add this function immediately after `_error_as_assistant_message` (which stays unchanged in `router.py`):

```python
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
    error_obj = parsed.get("error")
    if isinstance(error_obj, dict) and isinstance(error_obj.get("message"), str):
        error_obj["message"] = note + error_obj["message"]
        return json.dumps(parsed).encode()
    return note.encode() + error_body
```

- [ ] **Step 5: Verify with a full syntax + smoke check**

```bash
cd /Users/imranqureshi/git/coding-model-router
python3 -c "import ast; ast.parse(open('router.py').read())"
ROUTER_CONFIG=/tmp/empty_router_config.json ROUTER_PORT=18771 python3 router.py &
sleep 1
curl -sf http://127.0.0.1:18771/health && echo " OK"
curl -sf http://127.0.0.1:18771/v1/models && echo " OK"
kill %1
```

- [ ] **Step 6: Commit**

```bash
git add router.py
git commit -m "Wire Strategy A/B context enforcement; fix passthrough model-id override and annotate fallback-route errors"
```

---

### Task 9: Add new optional fields to `models.json`

**Files:**
- Modify: `models.json`

**Interfaces:**
- Consumes: none.
- Produces: `tokenizer_model`/`tokenizer_safety_margin` on Qwen `bedrock_models` entries; `claude_model_pattern` on every `anthropic_models` tier entry (installer emits it into every generated route regardless of backend — see Task 10).

- [ ] **Step 1: Add `tokenizer_model` + `tokenizer_safety_margin` to the two Qwen `bedrock_models` entries**

Old (`models.json:84-100`):
```json
  "bedrock_models": [
    {
      "id":               "qwen.qwen3-coder-30b-a3b-v1:0",
      "context_window":   262144,
      "max_output_tokens": 32768,
      "api_type":         "openai",
      "price_per_mtok":   0.15,
      "reason":           "Best Bedrock coding model (default)"
    },
    {
      "id":               "qwen.qwen3-coder-next",
      "context_window":   262144,
      "max_output_tokens": 65536,
      "api_type":         "openai",
      "price_per_mtok":   0.50,
      "reason":           "Next-gen Qwen Coder on Bedrock"
    },
```

New:
```json
  "bedrock_models": [
    {
      "id":               "qwen.qwen3-coder-30b-a3b-v1:0",
      "context_window":   262144,
      "max_output_tokens": 32768,
      "api_type":         "openai",
      "price_per_mtok":   0.15,
      "tokenizer_model":  "Qwen/Qwen3-Coder-30B-A3B-Instruct",
      "tokenizer_safety_margin": 4096,
      "reason":           "Best Bedrock coding model (default)"
    },
    {
      "id":               "qwen.qwen3-coder-next",
      "context_window":   262144,
      "max_output_tokens": 65536,
      "api_type":         "openai",
      "price_per_mtok":   0.50,
      "tokenizer_model":  "Qwen/Qwen3-Coder-30B-A3B-Instruct",
      "tokenizer_safety_margin": 4096,
      "reason":           "Next-gen Qwen Coder on Bedrock"
    },
```

(The two Claude-on-Bedrock entries below are left unchanged — `api_type: "anthropic"` routes always use Strategy B, no tokenizer field needed.)

- [ ] **Step 2: Add `claude_model_pattern` to every `anthropic_models` tier entry**

Old (`models.json:119-124`):
```json
  "anthropic_models": [
    {"tier": "haiku",  "id": "claude-haiku-4-5-20251001", "max_output_tokens": null,  "price_per_mtok": 1.0},
    {"tier": "sonnet", "id": "claude-sonnet-4-6",          "max_output_tokens": null,  "price_per_mtok": 3.0},
    {"tier": "opus",   "id": "claude-opus-4-8",            "max_output_tokens": 65536, "price_per_mtok": 5.0},
    {"tier": "fable",  "id": "claude-fable-5",             "max_output_tokens": 65536, "price_per_mtok": 10.0}
  ],
```

New:
```json
  "anthropic_models": [
    {"tier": "haiku",  "id": "claude-haiku-4-5-20251001", "claude_model_pattern": "^claude-haiku-",      "max_output_tokens": null,  "price_per_mtok": 1.0},
    {"tier": "sonnet", "id": "claude-sonnet-4-6",          "claude_model_pattern": "^claude-sonnet(-|$)", "max_output_tokens": null,  "price_per_mtok": 3.0},
    {"tier": "opus",   "id": "claude-opus-4-8",            "claude_model_pattern": "^claude-opus-",       "max_output_tokens": 65536, "price_per_mtok": 5.0},
    {"tier": "fable",  "id": "claude-fable-5",             "claude_model_pattern": "^claude-fable-",      "max_output_tokens": 65536, "price_per_mtok": 10.0}
  ],
```

- [ ] **Step 3: Verify the file is still valid JSON**

```bash
cd /Users/imranqureshi/git/coding-model-router
python3 -c "import json; json.load(open('models.json'))" && echo "valid JSON"
```

- [ ] **Step 4: Commit**

```bash
git add models.json
git commit -m "Add tokenizer_model and claude_model_pattern fields to models.json"
```

---

### Task 10: Update `install-model-router.sh` — emit new fields, bundle new modules, add `transformers` dependency

**Files:**
- Modify: `install-model-router.sh`

**Interfaces:**
- Consumes: `models.json`'s new `tokenizer_model`, `tokenizer_safety_margin`, `claude_model_pattern` fields (Task 9).
- Produces: `router_config.json` routes carrying `claude_model_pattern` (all routes) and `tokenizer_model`/`backend_max_context_tokens`/`reserved_output_tokens`/`tokenizer_safety_margin` (Qwen Bedrock routes only).

- [ ] **Step 1: Emit the new shell variables in `_init_model_config()`**

Old (`install-model-router.sh:320-341`):
```python
bm = d.get('bedrock_models', [])
print(f"BEDROCK_MODEL_COUNT={len(bm)}")
for i, m in enumerate(bm):
    k = vk(m['id'])
    print(f"BEDROCK_MODEL_{i}={shq(m['id'])}")
    print(f"BEDROCK_REASON_{i}={shq(m.get('reason', ''))}")
    print(f"BEDROCK_CTX_{k}={m.get('context_window', '')}")
    v = m.get('max_output_tokens')
    print(f"BEDROCK_MAXOUT_{k}={v if v is not None else ''}")
    print(f"BEDROCK_APITYPE_{k}={shq(m.get('api_type', 'anthropic'))}")
    p = m.get('price_per_mtok', 0)
    print(f"BEDROCK_PRICE_{k}={p}")

for m in d.get('anthropic_models', []):
    t = m['tier'].upper()
    print(f"TIER_MODEL_{t}={shq(m['id'])}")
    v = m.get('max_output_tokens')
    print(f"TIER_MAXOUT_{t}={v if v is not None else ''}")
    if v:
        print(f"ANTHROPIC_MAXOUT_{vk(m['id'])}={v}")
    p = m.get('price_per_mtok', 0)
    print(f"TIER_PRICE_{t}={p}")
```

New:
```python
bm = d.get('bedrock_models', [])
print(f"BEDROCK_MODEL_COUNT={len(bm)}")
for i, m in enumerate(bm):
    k = vk(m['id'])
    print(f"BEDROCK_MODEL_{i}={shq(m['id'])}")
    print(f"BEDROCK_REASON_{i}={shq(m.get('reason', ''))}")
    print(f"BEDROCK_CTX_{k}={m.get('context_window', '')}")
    v = m.get('max_output_tokens')
    print(f"BEDROCK_MAXOUT_{k}={v if v is not None else ''}")
    print(f"BEDROCK_APITYPE_{k}={shq(m.get('api_type', 'anthropic'))}")
    p = m.get('price_per_mtok', 0)
    print(f"BEDROCK_PRICE_{k}={p}")
    print(f"BEDROCK_TOKENIZER_{k}={shq(m.get('tokenizer_model') or '')}")
    print(f"BEDROCK_MARGIN_{k}={m.get('tokenizer_safety_margin', 4096)}")

for m in d.get('anthropic_models', []):
    t = m['tier'].upper()
    print(f"TIER_MODEL_{t}={shq(m['id'])}")
    v = m.get('max_output_tokens')
    print(f"TIER_MAXOUT_{t}={v if v is not None else ''}")
    if v:
        print(f"ANTHROPIC_MAXOUT_{vk(m['id'])}={v}")
    p = m.get('price_per_mtok', 0)
    print(f"TIER_PRICE_{t}={p}")
    print(f"TIER_PATTERN_{t}={shq(m.get('claude_model_pattern') or '')}")
```

- [ ] **Step 2: Add three new shell helper functions**

Old (`install-model-router.sh:476-479`):
```bash
tier_price() {  # price per million tokens for an Anthropic tier (haiku/sonnet/opus/fable); 0 if unknown
  local T; T="$(printf '%s' "$1" | tr '[:lower:]' '[:upper:]')"
  eval "printf '%s' \"\${TIER_PRICE_${T}:-0}\""
}
```

New (add the three new functions immediately after):
```bash
tier_price() {  # price per million tokens for an Anthropic tier (haiku/sonnet/opus/fable); 0 if unknown
  local T; T="$(printf '%s' "$1" | tr '[:lower:]' '[:upper:]')"
  eval "printf '%s' \"\${TIER_PRICE_${T}:-0}\""
}

bedrock_model_tokenizer() {  # HF tokenizer id for a Bedrock model; empty if not configured
  local k; k="$(_model_key "$1")"
  eval "printf '%s' \"\${BEDROCK_TOKENIZER_${k}:-}\""
}

bedrock_model_margin() {  # tokenizer safety margin (tokens) for a Bedrock model; 4096 if unset
  local k; k="$(_model_key "$1")"
  eval "printf '%s' \"\${BEDROCK_MARGIN_${k}:-4096}\""
}

tier_pattern() {  # claude_model_pattern regex for an Anthropic tier; empty if unknown
  local T; T="$(printf '%s' "$1" | tr '[:lower:]' '[:upper:]')"
  eval "printf '%s' \"\${TIER_PATTERN_${T}:-}\""
}
```

- [ ] **Step 3: Rewrite `build_route()` to emit `claude_model_pattern` on every branch and tokenizer fields on Qwen Bedrock routes**

Old (`install-model-router.sh:989-1045`):
```bash
build_route() {
  # $1=tier(haiku|sonnet|opus|fable)  $2=claude_model_name
  # Config shape: {tier, claude_model, url, model, auth}
  #   url   — full upstream endpoint (explicit, no construction in router)
  #   model — upstream model name (replaces claude_model in request body)
  #   auth  — none | passthrough | aws
  local be bid tier_name="$1" claude_model="$2"
  be="$(tier_backend "$1")"
  bid="$(tier_bedrock "$1")"
  if [ "$be" = "local" ]; then
    # max_tokens: written into the route so the router raises Claude Code's conservative cap
    # to the local model's actual context window. Omitted when the user chose "none" (vllm-mlx
    # uses its built-in default and the router won't override).
    #
    # chat_template_kwargs: Qwen3 models emit a lengthy reasoning chain by default; setting
    # enable_thinking=false in the tokenizer template skips it entirely so responses are fast.
    local _ctk=""
    if printf '%s' "$MODEL_ID" | grep -qi 'qwen3'; then
      _ctk=', "chat_template_kwargs": {"enable_thinking": false}'
    fi
    if [ "$MLX_MAX_TOKENS" = "none" ]; then
      printf '    {"tier": "%s", "claude_model": "%s", "url": "http://localhost:%s/v1/messages", "model": "%s", "auth": "none", "price_per_mtok": 0%s}' \
        "$tier_name" "$claude_model" "$MLX_PORT" "$MODEL_ID" "$_ctk"
    else
      printf '    {"tier": "%s", "claude_model": "%s", "url": "http://localhost:%s/v1/messages", "model": "%s", "auth": "none", "max_tokens": %s, "price_per_mtok": 0%s}' \
        "$tier_name" "$claude_model" "$MLX_PORT" "$MODEL_ID" "$MLX_MAX_TOKENS" "$_ctk"
    fi
  elif [ "$be" = "bedrock" ]; then
    # Qwen models (prefix "qwen.") use the OpenAI Chat Completions wire format on Bedrock Mantle;
    # Claude models use the Anthropic Messages format. The two paths are distinct URL namespaces.
    # Qwen models support the full context window as max output tokens — write max_tokens into
    # the route so the router overrides Claude Code's conservative 8K cap. Claude Bedrock models
    # have their own (lower) output ceiling and don't need the override.
    local _bprice; _bprice="$(bedrock_model_price "$bid")"
    local _bctx; _bctx="$(bedrock_model_ctx "$bid")"
    local _bmaxout; _bmaxout="$(bedrock_model_max_output "$bid")"
    # For Qwen models, include context_window for dynamic token calculation
    # and max_tokens to override Claude Code's conservative 8K cap
    if printf '%s' "$bid" | grep -qi '^qwen'; then
      if [ -n "$_bctx" ]; then
        printf '    {"tier": "%s", "claude_model": "%s", "url": "https://bedrock-mantle.%s.api.aws/v1/chat/completions", "model": "%s", "auth": "aws", "aws_region": "%s", "api_type": "openai", "price_per_mtok": %s, "context_window": %s, "max_tokens": %s}' \
          "$tier_name" "$claude_model" "$AWS_REGION" "$bid" "$AWS_REGION" "$_bprice" "$_bctx" "$_bmaxout"
      else
        printf '    {"tier": "%s", "claude_model": "%s", "url": "https://bedrock-mantle.%s.api.aws/v1/chat/completions", "model": "%s", "auth": "aws", "aws_region": "%s", "api_type": "openai", "price_per_mtok": %s, "max_tokens": %s}' \
          "$tier_name" "$claude_model" "$AWS_REGION" "$bid" "$AWS_REGION" "$_bprice" "$_bmaxout"
      fi
    else
      printf '    {"tier": "%s", "claude_model": "%s", "url": "https://bedrock-mantle.%s.api.aws/anthropic/v1/messages", "model": "%s", "auth": "aws", "aws_region": "%s", "price_per_mtok": %s}' \
        "$tier_name" "$claude_model" "$AWS_REGION" "$bid" "$AWS_REGION" "$_bprice"
    fi
  else
    # anthropic — passthrough Authorization header from Claude Code.
    local _aprice; _aprice="$(tier_price "$tier_name")"
    printf '    {"tier": "%s", "claude_model": "%s", "url": "https://api.anthropic.com/v1/messages", "model": "%s", "auth": "passthrough", "price_per_mtok": %s}' \
      "$tier_name" "$claude_model" "$claude_model" "$_aprice"
  fi
}
```

New:
```bash
build_route() {
  # $1=tier(haiku|sonnet|opus|fable)  $2=claude_model_name
  # Config shape: {tier, claude_model, claude_model_pattern, url, model, auth}
  #   url   — full upstream endpoint (explicit, no construction in router)
  #   model — upstream model name (replaces claude_model in request body)
  #   auth  — none | passthrough | aws
  #   claude_model_pattern — optional regex; lets the route keep matching future
  #     Claude Code model-id version bumps without a config edit (see route_config.py).
  local be bid tier_name="$1" claude_model="$2"
  be="$(tier_backend "$1")"
  bid="$(tier_bedrock "$1")"

  local _pattern; _pattern="$(tier_pattern "$tier_name")"
  local _cmp=""
  [ -n "$_pattern" ] && _cmp=", \"claude_model_pattern\": \"$_pattern\""

  if [ "$be" = "local" ]; then
    # max_tokens: written into the route so the router raises Claude Code's conservative cap
    # to the local model's actual context window. Omitted when the user chose "none" (vllm-mlx
    # uses its built-in default and the router won't override).
    #
    # chat_template_kwargs: Qwen3 models emit a lengthy reasoning chain by default; setting
    # enable_thinking=false in the tokenizer template skips it entirely so responses are fast.
    local _ctk=""
    if printf '%s' "$MODEL_ID" | grep -qi 'qwen3'; then
      _ctk=', "chat_template_kwargs": {"enable_thinking": false}'
    fi
    if [ "$MLX_MAX_TOKENS" = "none" ]; then
      printf '    {"tier": "%s", "claude_model": "%s", "url": "http://localhost:%s/v1/messages", "model": "%s", "auth": "none", "price_per_mtok": 0%s%s}' \
        "$tier_name" "$claude_model" "$MLX_PORT" "$MODEL_ID" "$_ctk" "$_cmp"
    else
      printf '    {"tier": "%s", "claude_model": "%s", "url": "http://localhost:%s/v1/messages", "model": "%s", "auth": "none", "max_tokens": %s, "price_per_mtok": 0%s%s}' \
        "$tier_name" "$claude_model" "$MLX_PORT" "$MODEL_ID" "$MLX_MAX_TOKENS" "$_ctk" "$_cmp"
    fi
  elif [ "$be" = "bedrock" ]; then
    # Qwen models (prefix "qwen.") use the OpenAI Chat Completions wire format on Bedrock Mantle;
    # Claude models use the Anthropic Messages format. The two paths are distinct URL namespaces.
    # Qwen models support the full context window as max output tokens — write max_tokens into
    # the route so the router overrides Claude Code's conservative 8K cap. Claude Bedrock models
    # have their own (lower) output ceiling and don't need the override.
    local _bprice; _bprice="$(bedrock_model_price "$bid")"
    local _bctx; _bctx="$(bedrock_model_ctx "$bid")"
    local _bmaxout; _bmaxout="$(bedrock_model_max_output "$bid")"
    # For Qwen models, include context_window for dynamic token calculation
    # and max_tokens to override Claude Code's conservative 8K cap
    if printf '%s' "$bid" | grep -qi '^qwen'; then
      # tokenizer_model enables Strategy A (preflight compression via context_manager.py)
      # instead of the character-heuristic Strategy B — see router.py docstring.
      local _btok; _btok="$(bedrock_model_tokenizer "$bid")"
      local _tok_json=""
      if [ -n "$_btok" ] && [ -n "$_bctx" ] && [ -n "$_bmaxout" ]; then
        local _bmargin; _bmargin="$(bedrock_model_margin "$bid")"
        _tok_json=", \"tokenizer_model\": \"$_btok\", \"backend_max_context_tokens\": $_bctx, \"reserved_output_tokens\": $_bmaxout, \"tokenizer_safety_margin\": $_bmargin"
      fi
      if [ -n "$_bctx" ]; then
        printf '    {"tier": "%s", "claude_model": "%s", "url": "https://bedrock-mantle.%s.api.aws/v1/chat/completions", "model": "%s", "auth": "aws", "aws_region": "%s", "api_type": "openai", "price_per_mtok": %s, "context_window": %s, "max_tokens": %s%s%s}' \
          "$tier_name" "$claude_model" "$AWS_REGION" "$bid" "$AWS_REGION" "$_bprice" "$_bctx" "$_bmaxout" "$_tok_json" "$_cmp"
      else
        printf '    {"tier": "%s", "claude_model": "%s", "url": "https://bedrock-mantle.%s.api.aws/v1/chat/completions", "model": "%s", "auth": "aws", "aws_region": "%s", "api_type": "openai", "price_per_mtok": %s, "max_tokens": %s%s}' \
          "$tier_name" "$claude_model" "$AWS_REGION" "$bid" "$AWS_REGION" "$_bprice" "$_bmaxout" "$_cmp"
      fi
    else
      printf '    {"tier": "%s", "claude_model": "%s", "url": "https://bedrock-mantle.%s.api.aws/anthropic/v1/messages", "model": "%s", "auth": "aws", "aws_region": "%s", "price_per_mtok": %s%s}' \
        "$tier_name" "$claude_model" "$AWS_REGION" "$bid" "$AWS_REGION" "$_bprice" "$_cmp"
    fi
  else
    # anthropic — passthrough Authorization header from Claude Code.
    local _aprice; _aprice="$(tier_price "$tier_name")"
    printf '    {"tier": "%s", "claude_model": "%s", "url": "https://api.anthropic.com/v1/messages", "model": "%s", "auth": "passthrough", "price_per_mtok": %s%s}' \
      "$tier_name" "$claude_model" "$claude_model" "$_aprice" "$_cmp"
  fi
}
```

- [ ] **Step 4: Update the bootstrap file lists to bundle the new modules**

Old (`install-model-router.sh:80`):
```bash
    for file in install-model-router.sh start-model-router.sh stop-model-router.sh uninstall-model-router.sh router.py models.json mcp-local.json; do
```

New:
```bash
    for file in install-model-router.sh start-model-router.sh stop-model-router.sh uninstall-model-router.sh router.py constants.py tool_sanitizer.py aws_auth.py bedrock_client.py route_config.py message_translator.py tokenizer.py context_manager.py usage_stats.py stream_converter.py models.json mcp-local.json; do
```

Old (`install-model-router.sh:104`):
```bash
    cp "$SCRIPT_DIR/"*.sh "$SCRIPT_DIR/router.py" "$SCRIPT_DIR/models.json" "$SCRIPT_DIR/mcp-local.json" "$DIR/"
```

New:
```bash
    cp "$SCRIPT_DIR/"*.sh "$SCRIPT_DIR/router.py" "$SCRIPT_DIR/constants.py" "$SCRIPT_DIR/tool_sanitizer.py" "$SCRIPT_DIR/aws_auth.py" "$SCRIPT_DIR/bedrock_client.py" "$SCRIPT_DIR/route_config.py" "$SCRIPT_DIR/message_translator.py" "$SCRIPT_DIR/tokenizer.py" "$SCRIPT_DIR/context_manager.py" "$SCRIPT_DIR/usage_stats.py" "$SCRIPT_DIR/stream_converter.py" "$SCRIPT_DIR/models.json" "$SCRIPT_DIR/mcp-local.json" "$DIR/"
```

- [ ] **Step 5: Add `transformers` to the pip install list when any tier routes to Bedrock**

Old (`install-model-router.sh:951` and `:969`, both occurrences):
```bash
  _extra_pip=""; any_bedrock && _extra_pip="boto3 openai"
```

New (both occurrences):
```bash
  _extra_pip=""; any_bedrock && _extra_pip="boto3 openai transformers"
```

`transformers` is only needed for `tokenizer.py`'s exact token counting on Bedrock/Qwen routes (`tokenizer.py` imports it lazily, inside `_load_tokenizer`, so this doesn't slow down local-only installs). It has no local-model-only use case, so it's gated behind `any_bedrock` exactly like `boto3`/`openai` already are.

- [ ] **Step 6: `start-model-router.sh` needs no changes**

Confirm this by inspection — `start-model-router.sh:128` invokes `"$VENV/bin/python" "$DIR/router.py"` directly. Since all the new modules live alongside `router.py` in the same `$DIR` (`~/model-router/`), Python automatically resolves `import constants`, `import route_config`, etc. via the script's own directory on `sys.path` — no explicit `PYTHONPATH` or launcher change needed.

```bash
grep -n "router.py" /Users/imranqureshi/git/coding-model-router/start-model-router.sh
```
Expected: only the existing single invocation line; no further edits required.

- [ ] **Step 7: Verify the installer's shell syntax**

```bash
cd /Users/imranqureshi/git/coding-model-router
bash -n install-model-router.sh && echo "syntax OK"
```

- [ ] **Step 8: Commit**

```bash
git add install-model-router.sh
git commit -m "install-model-router.sh: emit tokenizer/pattern fields, bundle new modules, add transformers dep"
```

---

### Task 11: Update `CLAUDE.md` documentation

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update the "Source files (this repo)" table**

Old:
```
| File | Role |
|---|---|
| `install-model-router.sh` | Interactive installer — detects auth, picks model, installs venv, writes config, patches `~/.zshrc` |
| `router.py` | FastAPI proxy — routes by model name, handles auth strategies, translates OpenAI↔Anthropic wire formats |
| `models.json` | Model catalog — context windows, RAM thresholds, Bedrock IDs, tier defaults. Edit here; re-run installer to apply. |
| `start-model-router.sh` | Launches vllm-mlx (optional) then router.py; waits for vllm-mlx readiness before starting router |
| `stop-model-router.sh` | Kills running processes by PID file |
| `test-vllm-mlx.py` | Smoke tests a live vllm-mlx server (health, basic reply, tool call, tool result roundtrip) |
| `test-bedrock.py` | Smoke tests Bedrock routing through the running router (same four tests) |
| `mcp-local.json` | MCP config passed to `claude --mcp-config` when using `claude-router` |
```

New:
```
| File | Role |
|---|---|
| `install-model-router.sh` | Interactive installer — detects auth, picks model, installs venv, writes config, patches `~/.zshrc` |
| `router.py` | FastAPI app + request orchestration — routes by model name, dispatches by auth strategy. Supporting logic lives in the sibling modules below. |
| `constants.py` | Shared regexes, header allowlists, retry/backoff tuning constants |
| `tool_sanitizer.py` | vllm-mlx tool-name sanitization (local routes only) |
| `aws_auth.py` | SigV4 signing for Bedrock (`_sign_bedrock`, `_SigV4Auth`) |
| `bedrock_client.py` | Bedrock dispatch pacing, throttle detection, retry-with-backoff |
| `route_config.py` | Route loading + lookup, including `claude_model_pattern` regex fallback |
| `message_translator.py` | Anthropic↔OpenAI request/response translation |
| `tokenizer.py` | HuggingFace tokenizer loading + exact token counting (Bedrock/Qwen routes with `tokenizer_model` configured) |
| `context_manager.py` | Preflight context-budget enforcement: tool-result compression + oldest-message dropping |
| `usage_stats.py` | Cumulative token stats + the stdout "savings ticker" |
| `stream_converter.py` | SSE streaming translation (OpenAI SDK stream → Anthropic SSE) |
| `models.json` | Model catalog — context windows, RAM thresholds, Bedrock IDs, tier defaults. Edit here; re-run installer to apply. |
| `start-model-router.sh` | Launches vllm-mlx (optional) then router.py; waits for vllm-mlx readiness before starting router |
| `stop-model-router.sh` | Kills running processes by PID file |
| `test-vllm-mlx.py` | Smoke tests a live vllm-mlx server (health, basic reply, tool call, tool result roundtrip) |
| `test-bedrock.py` | Smoke tests Bedrock routing through the running router (same four tests) |
| `mcp-local.json` | MCP config passed to `claude --mcp-config` when using `claude-router` |
```

- [ ] **Step 2: Update the "How the router enforces max_tokens" section to describe the dual strategy**

Old:
```
### How the router enforces max_tokens

Claude Code sends `max_tokens=128000` in every request (result of `min(200000, 128000)`). The router then applies two caps before the request reaches Bedrock:

1. **Dynamic context cap** — estimates input token count from character length, subtracts from `context_window` (262144 for Qwen), and caps `max_tokens` to the remainder. Prevents Bedrock from rejecting the request for context overflow.
2. **Route ceiling** — hard cap from `router_config.json` (`max_tokens: 65536` for sonnet, `32768` for haiku). Matches the model's actual output limit.

Effective max output per turn = `min(route_ceiling, remaining_context_after_input)`.
```

New:
```
### How the router enforces max_tokens

Claude Code sends `max_tokens=128000` in every request (result of `min(200000, 128000)`). The router uses one of two mutually-exclusive strategies, chosen per-route:

**Strategy A — tokenizer-based (routes with `tokenizer_model` configured):** translates the request to OpenAI format, counts exact tokens via the model's own HuggingFace tokenizer + chat template (`tokenizer.py`), and if over `backend_max_context_tokens - reserved_output_tokens - tokenizer_safety_margin`, compresses oversized tool results (head+tail truncation) and drops oldest message groups until it fits (`context_manager.py`), before ever sending the request upstream.

**Strategy B — character-heuristic (all other routes, including local/vllm-mlx):** estimates input token count from character length (4 chars ≈ 1 token, 1.20× safety multiplier), subtracts from `context_window`, and caps `max_tokens` to the remainder, minus a fixed `_TOKEN_ESTIMATE_SAFETY_BUFFER` (100 tokens) to absorb estimation imprecision on dense content like JSON/tool schemas.

Both strategies are additionally bounded by the **route ceiling** — a hard cap from `router_config.json` (`max_tokens: 65536` for sonnet, `32768` for haiku) matching the model's actual output limit. A reactive halving-retry loop (below) remains as a last-resort safety net for both strategies.
```

- [ ] **Step 3: Update the route config schema description to mention new optional fields**

Add a note after the existing "Context overflow retry loop" section:

```
### Route config schema additions

Two new optional route fields (see `router.py`'s module docstring for the full list):

- `claude_model_pattern` — regex checked on an exact `claude_model` lookup miss, so a route keeps matching future Claude Code model-id version bumps (e.g. `claude-sonnet-4-6` → `claude-sonnet-5`) without a `models.json` edit + reinstall.
- `tokenizer_model` / `backend_max_context_tokens` / `reserved_output_tokens` / `tokenizer_safety_margin` — activate Strategy A (above) for a route. Currently set on the two Qwen Bedrock entries in `models.json`.
```

- [ ] **Step 4: Verify the doc renders sensibly**

```bash
cd /Users/imranqureshi/git/coding-model-router
grep -c "^###\|^##" CLAUDE.md  # sanity check headers are still well-formed
```

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md
git commit -m "Document module split, dual context-enforcement strategy, and new route config fields"
```

---

### Task 12: Full verification pass

**Files:** none (verification only).

- [ ] **Step 1: Run the full pytest suite**

```bash
cd /Users/imranqureshi/git/coding-model-router
.venv-dev/bin/python -m pytest tests/ -v
```
Expected: all tests across `test_bedrock_client.py`, `test_route_config.py`, `test_context_manager.py` PASS (33 tests total).

- [ ] **Step 2: Run a full local install in no-local-model mode and confirm `router_config.json` shape**

```bash
cd /Users/imranqureshi/git/coding-model-router
USE_LOCAL_MODELS=0 bash install-model-router.sh --mode B
python3 -c "
import json
cfg = json.load(open('$HOME/model-router/router_config.json'))
for r in cfg['routes']:
    print(r['tier'], r['auth'], r.get('claude_model_pattern'), r.get('tokenizer_model'))
"
```
Expected: four routes (haiku/sonnet/opus/fable), each with a non-empty `claude_model_pattern`; the Bedrock-backed tier(s) additionally show a `tokenizer_model` value (e.g. `Qwen/Qwen3-Coder-30B-A3B-Instruct`) if that tier's default backend is `bedrock` with a Qwen model.

- [ ] **Step 3: Start the router and confirm health/models**

```bash
start-model-router &
sleep 3
curl -sf http://localhost:8771/health && echo " OK"
curl -sf http://localhost:8771/v1/models && echo " OK"
```
Expected: `{"status":"ok","routes":4} OK` and a model list containing all four configured `claude_model` ids.

- [ ] **Step 4: If AWS credentials are configured, run the existing Bedrock smoke test**

```bash
python3 test-bedrock.py --router-port 8771
```
Expected: all four tests (health, basic reply, tool call, tool result roundtrip) PASS — this exercises the new Strategy A context-budget path end-to-end if the sonnet/haiku tier routes to a Qwen Bedrock model with `tokenizer_model` set, and exercises the a1/a2 bug fixes under real throttling/streaming conditions.

- [ ] **Step 5: Stop the router and review the full diff**

```bash
stop-model-router
cd /Users/imranqureshi/git/coding-model-router
git log --oneline main..HEAD  # or the feature branch, if one was used
git diff main --stat
```
Confirm: no unintended files touched (`vllm_mlx-0.4.0-py3-none-any.whl`, `README.md` untouched unless intentionally updated), `.venv-dev/` not tracked, `__pycache__/` not tracked.

- [ ] **Step 6: Final commit (only if step 5 surfaces cleanup)**

If `git status` shows anything unexpected (e.g. a stray `__pycache__/` got staged in an earlier task), fix it now:
```bash
git rm -r --cached __pycache__ 2>/dev/null; git commit -m "Remove stray __pycache__ from version control" 2>/dev/null || true
```
Otherwise no commit is needed — this task is verification-only.

---

## Self-Review

**Spec coverage:**
- Module split mirroring the gateway ✅ (Tasks 1–8)
- Bug fixes a1 (throttle/overflow misclassification) ✅ Task 2, a2 (`stream.close()` awaitable safety) ✅ Task 7, a5 (passthrough model-id preservation) ✅ Task 8, a6 (fallback-error annotation) ✅ Task 8
- `_TOKEN_ESTIMATE_SAFETY_BUFFER` hardening constant ✅ Task 1 (constants.py) + wired in Task 8
- Tokenizer-based context budget pipeline (approved new capability) ✅ Task 5
- `claude_model_pattern` regex route fallback (approved new capability) ✅ Task 3, config ✅ Task 9, installer ✅ Task 10
- Explicitly NOT porting MongoDB/OIDC/OTel ✅ stated in Global Constraints, no task introduces them
- Preserve local-route support, tool sanitization, `chat_template_kwargs`, savings ticker, overflow-halving retry ✅ verified present in Task 8's rewritten `proxy_messages` and Task 6's `usage_stats.py`
- Avoid the gateway's `output_tokens` dead-variable regression ✅ Task 7 explicitly calls this out and preserves baseline's correct tracking
- Documentation updates ✅ Task 11
- Installer/config wiring ✅ Tasks 9–10

**Placeholder scan:** No "TBD"/"add error handling"/"similar to Task N" phrases — every step shows complete, copy-pasteable code or exact shell commands with expected output.

**Type/name consistency:** `find_route`, `ROUTES`, `CONFIG`, `CONFIG_PATH` (route_config.py) match their call sites in `router.py`/`usage_stats.py` across all tasks. `_record_tokens`'s signature (`upstream_model, in_tok, out_tok, price_per_mtok, backend_label, tier=""`) is identical everywhere it's called (Tasks 6, 7, 8). `_is_throttling_status`/`_throttle_backoff_seconds`/`_send_with_bedrock_retry` signatures match between `bedrock_client.py` (Task 2) and their call sites in `router.py` (Task 8).
