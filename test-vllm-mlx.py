#!/usr/bin/env python3
"""
test-vllm-mlx.py — smoke test a running vllm-mlx server.

Usage:
  # Start vllm-mlx in another terminal first:
  #   vllm-mlx serve mlx-community/Qwen3.5-9B-MLX-4bit --port 8770
  #
  python3 test-vllm-mlx.py [--port 8770] [--model mlx-community/Qwen3.5-9B-MLX-4bit]

Tests (in order):
  1. /health — server is up
  2. Basic message — correct stop_reason: end_turn (not "stop")
  3. Tool call — model returns a tool_use block (not JSON-in-text)
  4. Tool result roundtrip — model continues after a tool_result and produces end_turn
"""
import argparse
import json
import sys
import urllib.request
import urllib.error

parser = argparse.ArgumentParser()
parser.add_argument("--port",  type=int,  default=8770)
parser.add_argument("--model", default="mlx-community/Qwen3.5-9B-MLX-4bit")
parser.add_argument("--stream", action="store_true", help="use streaming (SSE)")
args = parser.parse_args()

BASE = f"http://localhost:{args.port}"
PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
failures = 0


def post(path, body, stream=False):
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        BASE + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            raw = r.read()
        if stream:
            # SSE: accumulate data: lines, parse last complete event
            events = []
            for line in raw.decode().splitlines():
                if line.startswith("data: "):
                    payload = line[6:]
                    if payload == "[DONE]":
                        break
                    try:
                        events.append(json.loads(payload))
                    except json.JSONDecodeError:
                        pass
            return events
        return json.loads(raw)
    except urllib.error.HTTPError as e:
        return {"error": e.read().decode(), "status": e.code}
    except Exception as e:
        return {"error": str(e)}


def check(label, resp, condition, detail=""):
    global failures
    ok = condition(resp)
    status = PASS if ok else FAIL
    print(f"  {status}  {label}")
    if not ok:
        failures += 1
        print(f"       response: {json.dumps(resp, indent=2)[:400]}")
        if detail:
            print(f"       expected: {detail}")


print(f"\nvllm-mlx smoke test  ({BASE}  model={args.model})\n")

# ── Test 1: health ─────────────────────────────────────────────────────────
print("1. Health check")
resp = None
try:
    with urllib.request.urlopen(BASE + "/health", timeout=5) as r:
        resp = json.loads(r.read())
    check("server is up", resp, lambda r: r.get("status") in ("ok", "healthy"))
except Exception as e:
    print(f"  {FAIL}  server is up — {e}")
    print("\nStart vllm-mlx first:")
    print(f"  vllm-mlx serve {args.model} --port {args.port}")
    sys.exit(1)

# ── Test 2: basic response + stop_reason ──────────────────────────────────
print("\n2. Basic message (stop_reason must be 'end_turn')")
resp = post("/v1/messages", {
    "model": args.model,
    "max_tokens": 64,
    "messages": [{"role": "user", "content": "Reply with exactly: hello"}],
})
check("stop_reason == end_turn", resp,
      lambda r: r.get("stop_reason") == "end_turn",
      "stop_reason: end_turn  (not 'stop' — which breaks the Claude Code agent loop)")
check("content block present", resp,
      lambda r: bool(r.get("content")))

# ── Test 3: tool call ─────────────────────────────────────────────────────
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
    "model": args.model,
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

# ── Test 4: tool result roundtrip ─────────────────────────────────────────
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
        "model": args.model,
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
    print(f"  (skipped — no tool_use block returned in test 3)")

# ── Summary ───────────────────────────────────────────────────────────────
print()
if failures == 0:
    print("✓ All tests passed — vllm-mlx tool loop works correctly\n")
else:
    print(f"✗ {failures} test(s) failed\n")
    sys.exit(1)
