#!/usr/bin/env python3
"""
test-bedrock.py — smoke test Bedrock Mantle routing via the model-router proxy.

The router handles SigV4 signing; this script just POSTs to the router on the
model that is configured for Bedrock (auth: "aws").

Usage:
  # Ensure model-router is running with a Bedrock route configured:
  #   start-model-router
  #
  python3 test-bedrock.py [--router-port 8771] [--model claude-sonnet-4-6]
  python3 test-bedrock.py --model claude-opus-4-8   # if opus routes to Bedrock

Tests (in order):
  1. /health — router is up
  2. Basic message — correct stop_reason: end_turn
  3. Tool call — model returns a tool_use block (not JSON-in-text)
  4. Tool result roundtrip — model continues after tool_result and produces end_turn
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Args / config
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--router-port", type=int, default=8771)
parser.add_argument(
    "--model",
    default=None,
    help="Claude model name that routes to Bedrock (auto-detected from config if omitted)",
)
args = parser.parse_args()


def _find_bedrock_model() -> str:
    """Return the claude_model for the first Bedrock (auth: aws) route in the router config."""
    config_path = Path(
        os.environ.get("ROUTER_CONFIG", Path.home() / "model-router" / "router_config.json")
    )
    try:
        with open(config_path) as f:
            config = json.load(f)
        for route in config.get("routes", []):
            if route.get("auth") == "aws":
                return route["claude_model"]
    except FileNotFoundError:
        pass
    return "claude-sonnet-4-6"


MODEL = args.model or _find_bedrock_model()
BASE = f"http://localhost:{args.router_port}"

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
failures = 0

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def post(path: str, body: dict) -> dict:
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        BASE + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return {"error": e.read().decode(), "status": e.code}
    except Exception as e:
        return {"error": str(e)}


def check(label: str, resp: dict, condition, detail: str = "") -> None:
    global failures
    ok = condition(resp)
    status = PASS if ok else FAIL
    print(f"  {status}  {label}")
    if not ok:
        failures += 1
        print(f"       response: {json.dumps(resp, indent=2)[:500]}")
        if detail:
            print(f"       expected: {detail}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

print(f"\nBedrock smoke test  ({BASE}  model={MODEL})\n")

# ── Test 1: router health ────────────────────────────────────────────────────
print("1. Router health check")
try:
    with urllib.request.urlopen(BASE + "/health", timeout=5) as r:
        resp = json.loads(r.read())
    check("router is up", resp, lambda r: r.get("status") == "ok")
except Exception as e:
    print(f"  {FAIL}  router is up — {e}")
    print("\nStart model-router first:  start-model-router")
    sys.exit(1)

# ── Test 2: basic response ───────────────────────────────────────────────────
print("\n2. Basic message (stop_reason must be 'end_turn')")
resp = post("/v1/messages", {
    "model": MODEL,
    "max_tokens": 64,
    "messages": [{"role": "user", "content": "Reply with exactly: hello"}],
})
check("stop_reason == end_turn", resp,
      lambda r: r.get("stop_reason") == "end_turn",
      "stop_reason: end_turn")
check("content block present", resp,
      lambda r: bool(r.get("content")))

# ── Test 3: tool call ────────────────────────────────────────────────────────
print("\n3. Tool call (model must return a tool_use block, not JSON-in-text)")
tools = [{
    "name": "get_weather",
    "description": "Get the current weather for a location.",
    "input_schema": {
        "type": "object",
        "properties": {"location": {"type": "string", "description": "City name"}},
        "required": ["location"],
    },
}]
resp = post("/v1/messages", {
    "model": MODEL,
    "max_tokens": 256,
    "tools": tools,
    "messages": [{"role": "user", "content": "What's the weather in Paris?"}],
})
check("stop_reason == tool_use", resp,
      lambda r: r.get("stop_reason") == "tool_use",
      "stop_reason: tool_use")
check("tool_use block in content", resp,
      lambda r: any(b.get("type") == "tool_use" for b in (r.get("content") or [])),
      "a content block with type=tool_use")

tool_use_block = next(
    (b for b in (resp.get("content") or []) if b.get("type") == "tool_use"), None
)
if tool_use_block:
    check("tool name correct", tool_use_block,
          lambda b: b.get("name") == "get_weather")
    check("tool input has location", tool_use_block,
          lambda b: "location" in (b.get("input") or {}))

# ── Test 4: tool result roundtrip ────────────────────────────────────────────
print("\n4. Tool result roundtrip (model must produce end_turn after tool_result)")
if tool_use_block:
    messages = [
        {"role": "user", "content": "What's the weather in Paris?"},
        {"role": "assistant", "content": resp.get("content", [])},
        {"role": "user", "content": [{
            "type": "tool_result",
            "tool_use_id": tool_use_block.get("id", ""),
            "content": "Sunny, 22°C",
        }]},
    ]
    resp2 = post("/v1/messages", {
        "model": MODEL,
        "max_tokens": 256,
        "tools": tools,
        "messages": messages,
    })
    check("stop_reason == end_turn after tool_result", resp2,
          lambda r: r.get("stop_reason") == "end_turn",
          "end_turn  (confirms tool loop works end-to-end)")
    check("response mentions temperature or weather", resp2,
          lambda r: any(
              "22" in str(b.get("text", "")) or "sunny" in str(b.get("text", "")).lower()
              for b in (r.get("content") or [])
          ))
else:
    print("  (skipped — no tool_use block returned in test 3)")

# ── Summary ──────────────────────────────────────────────────────────────────
print()
if failures == 0:
    print("✓ All tests passed — Bedrock Mantle routing works correctly\n")
else:
    print(f"✗ {failures} test(s) failed\n")
    sys.exit(1)
