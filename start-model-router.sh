#!/bin/bash
set -e

# start-model-router.sh — launch the local stack (vllm-mlx + router.py) in the foreground.
# Leave this running in its own terminal; use `claude-router` from another terminal.

DIR="$HOME/model-router"
if [ ! -f "$DIR/model-router.env" ]; then
  echo "ERROR: $DIR/model-router.env not found — run install-model-router first." >&2
  exit 1
fi
# shellcheck source=/dev/null
. "$DIR/model-router.env"

# Packages live in a dedicated venv inside the bundle (see install-model-router.sh).
VENV="$DIR/venv"
if [ ! -x "$VENV/bin/uvicorn" ]; then
  echo "ERROR: $VENV/bin/uvicorn not found — re-run install-model-router first." >&2
  exit 1
fi

ROUTER_LOG="${ROUTER_LOG_FILE:-$DIR/router.log}"
_session="===== model-router session started $(date '+%Y-%m-%d %H:%M:%S') ====="
printf '%s\n' "$_session" >> "$ROUTER_LOG"

MLX_PID=""
if [ "${USE_LOCAL_MODELS:-1}" = "1" ]; then
  if [ ! -x "$VENV/bin/vllm-mlx" ]; then
    echo "ERROR: $VENV/bin/vllm-mlx not found — re-run install-model-router first." >&2
    exit 1
  fi

  # Persistent log for the local model server.
  MLX_LOG="${MLX_LOG_FILE:-$DIR/mlx-lm.log}"
  printf '%s\n' "$_session" >> "$MLX_LOG"

  # VLLM_MLX_ENABLE_THINKING=false strips <think> token leaks from Qwen3-class models.
  # --kv-cache-quantization halves KV cache footprint (more headroom before OOM).
  # --stream-interval 4: batch 4 tokens per SSE chunk (fewer Python wakeups, better GPU throughput).
  # --timeout 600: raise from the 300s default to avoid disconnects on long generations.
  # Cache: use explicit --cache-memory-mb if PROMPT_CACHE_BYTES is set, otherwise default to
  # --cache-memory-percent 0.35 (35% of unified RAM, up from the 20% vllm-mlx default).
  MAX_TOK_ARGS=()
  [ -n "${MLX_MAX_TOKENS:-}" ] && [ "${MLX_MAX_TOKENS}" != "none" ] && MAX_TOK_ARGS=(--max-tokens "$MLX_MAX_TOKENS")
  PCB_ARGS=()
  if [ -n "${PROMPT_CACHE_BYTES:-}" ]; then
    PCB_ARGS=(--cache-memory-mb "$(( PROMPT_CACHE_BYTES / 1048576 ))")
  else
    PCB_ARGS=(--cache-memory-percent 0.35)
  fi

  # Resolve the tool-call parser for MODEL_ID (override via TOOL_CALL_PARSER env var).
  _detect_tool_parser() {
    local mcfg="$DIR/models.json"
    if [ -f "$mcfg" ]; then
      local _p
      _p="$(python3 -c "
import json, sys
d = json.load(open(sys.argv[1]))
m = next((x for x in d.get('local_models', []) if x.get('id') == sys.argv[2]), None)
if m and m.get('tool_parser'): print(m['tool_parser'])
" "$mcfg" "$MODEL_ID" 2>/dev/null)"
      if [ -n "$_p" ]; then echo "$_p"; return; fi
    fi
    local m; m="$(printf '%s' "$MODEL_ID" | tr '[:upper:]' '[:lower:]')"
    case "$m" in
      *qwen*)            echo "qwen" ;;
      *llama*)           echo "llama" ;;
      *mistral*)         echo "mistral" ;;
      *deepseek*)        echo "deepseek" ;;
      *hermes*|*nous*)   echo "hermes" ;;
      *kimi*|*moonshot*) echo "kimi" ;;
      *granite*)         echo "granite" ;;
      *glm4*)            echo "glm47" ;;
    esac
  }
  TOOL_PARSER_ARGS=()
  _parser="${TOOL_CALL_PARSER:-$(_detect_tool_parser)}"
  [ -n "$_parser" ] && TOOL_PARSER_ARGS=(--enable-auto-tool-choice --tool-call-parser "$_parser")

  echo "[1/2] vllm-mlx ($MODEL_ID) on :$MLX_PORT  (log: $MLX_LOG) ..."
  VLLM_MLX_ENABLE_THINKING=false \
  "$VENV/bin/vllm-mlx" serve "$MODEL_ID" --port "$MLX_PORT" \
    "${MAX_TOK_ARGS[@]}" "${PCB_ARGS[@]}" "${TOOL_PARSER_ARGS[@]}" \
    --kv-cache-quantization \
    --stream-interval 4 \
    --timeout 600 \
    >> "$MLX_LOG" 2>&1 &
  MLX_PID=$!

  echo "      waiting for vllm-mlx to be ready (first run may take several minutes to download the model)..."
  _attempt=0
  until curl -sf "http://127.0.0.1:$MLX_PORT/health" 2>/dev/null | grep -q '"healthy"\|"ok"'; do
    sleep 3; _attempt=$(( _attempt + 1 ))
    [ $(( _attempt % 10 )) -eq 0 ] && echo "      still loading... ($(( _attempt * 3 ))s)"
    if [ "$_attempt" -ge 100 ]; then
      echo "ERROR: vllm-mlx not ready after 300s — check $MLX_LOG" >&2
      kill "$MLX_PID" 2>/dev/null || true
      exit 1
    fi
  done
fi

# If any tier is routed to Bedrock, it authenticates via the AWS profile's SSO session. That
# session expires (typically hours), so check it up front and tell the user how to renew —
# non-fatal, since other tiers still work and the session can be refreshed while running.
if [ -n "${HAIKU_BEDROCK:-}${SONNET_BEDROCK:-}${OPUS_BEDROCK:-}${FABLE_BEDROCK:-}" ]; then
  echo "      Bedrock tier(s) active (region ${AWS_REGION:-?}${AWS_PROFILE_NAME:+, profile $AWS_PROFILE_NAME}):"
  # Show which actual Bedrock model each complexity tier runs (the tier's request key is just an
  # alias — e.g. medium is sent as claude-sonnet-4-6 but runs bedrock/qwen.qwen3-coder-next).
  for _pair in "low:${HAIKU_BEDROCK:-}" "medium:${SONNET_BEDROCK:-}" "high:${OPUS_BEDROCK:-}" "extreme:${FABLE_BEDROCK:-}"; do
    _lvl="${_pair%%:*}"; _bid="${_pair#*:}"
    if [ -n "$_bid" ]; then echo "        ${_lvl} → bedrock/${_bid}"; fi
  done
  if command -v aws >/dev/null 2>&1; then
    _awsargs=""; [ -n "${AWS_PROFILE_NAME:-}" ] && _awsargs="--profile $AWS_PROFILE_NAME"
    # shellcheck disable=SC2086
    if ! aws sts get-caller-identity $_awsargs >/dev/null 2>&1; then
      echo "      ⚠ No valid AWS session — Bedrock calls will fail. Run:  aws sso login${AWS_PROFILE_NAME:+ --profile $AWS_PROFILE_NAME}" >&2
    fi
  fi
fi

echo "[2/2] model-router proxy on :$ROUTER_PORT  (log: $ROUTER_LOG) ..."
ROUTER_CONFIG="$HOME/model-router/router_config.json" \
ROUTER_PORT="$ROUTER_PORT" \
AWS_PROFILE="${AWS_PROFILE_NAME:-}" \
"$VENV/bin/python" "$DIR/router.py" \
  2>>"$ROUTER_LOG" &
ROUTER_PID=$!

if [ "${USE_LOCAL_MODELS:-1}" = "1" ]; then
  echo "✓ Ready. Claude Code → router(:$ROUTER_PORT) → vllm-mlx(:$MLX_PORT) [local] / Bedrock / Anthropic"
  echo "  Logs: $MLX_LOG (vllm-mlx)  ·  $ROUTER_LOG (router)"
else
  echo "✓ Ready. Claude Code → router(:$ROUTER_PORT) → Bedrock / Anthropic"
  echo "  Logs: $ROUTER_LOG (router)"
fi
echo "  Stop from another terminal:  stop-model-router   (or: kill ${MLX_PID:+$MLX_PID }$ROUTER_PID)"
wait
