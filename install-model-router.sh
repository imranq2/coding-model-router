#!/bin/bash
set -euo pipefail

# install-model-router.sh — set up local GPU model routing for Claude Code.
#
# The install-model-router skill just runs this file — with NO flags. Everything is
# auto-detected: the auth mode (OAuth subscription vs API key), the local model (from RAM),
# free ports. A bare re-run reuses ~/model-router/model-router.env.
#
# All flags are OPTIONAL overrides:
#   --mode A|B            A = OAuth subscription (no API key), B = Anthropic API key
#                         (default: detected — A if a Claude login exists, else B if a key is set)
#   --model <hf-repo-id>  default: chosen from installed RAM, sized to leave KV-cache headroom
#                         (<24GB→Qwen3-8B, 24-47GB→Qwen3-14B, >=48GB→Qwen3.6-27B)
#   --mlx-port <port>     default 8770 (auto-bumped if busy)
#   --router-port <port> default 8771 (auto-bumped if busy)
#   --api-key <sk-ant-..> Mode B: API key (default: $ANTHROPIC_API_KEY)
# Per-tier backend, named by COMPLEXITY (low=haiku, medium=sonnet, high=opus, extreme=fable).
# Each takes 'local' (mlx-lm), 'claude' (Anthropic), or 'bedrock=<model-id>' (AWS Bedrock) —
# pass bare 'bedrock' in a terminal to pick the id from a list (led by qwen.qwen3-coder-30b-a3b-v1:0):
#   --low <backend>       background / simple tasks   (default: local)
#   --medium <backend>    daily coding                (default: claude)
#   --high <backend>      hard reasoning / planning   (default: claude)
#   --extreme <backend>   hardest / long-horizon      (default: claude)
#   --local <levels>      shortcut: mark these levels 'local' (comma list, or "all"/"none").
#                         Equivalent to --low local etc.; per-level flags override it.
#   --max-tokens <N>      max output tokens per local-model response. Defaults to 65536 for 256K
#                         context models, 32768 for 128K, 16384 for 40K and below.
#                         Passed as --max-tokens to vllm-mlx. Does not affect Bedrock/Claude tiers.
#   --prompt-cache-bytes <size>  passthrough to mlx_lm.server (e.g. 8GB). Caps the
#                         CROSS-REQUEST KV reuse cache (not a single request). Default: unset.
#   --auto-compact-window <tokens|auto>  CLAUDE_CODE_AUTO_COMPACT_WINDOW for claude-router — makes
#                         Claude Code compact within the LOCAL model's window. Default "auto" =
#                         the selected model's actual window (40K Qwen3-8B/14B, 256K others).
#   --autocompact-pct <1-100>  CLAUDE_AUTOCOMPACT_PCT_OVERRIDE — compact at this % of the window
#                         (default 55; lower = compact earlier, smaller KV cache, more RAM headroom).
#                         Reduce to 40 or lower on machines with ≤24GB RAM to prevent OOM errors.
#   --aws-region <region> region for any bedrock= tier (default us-east-1).
#   --aws-profile <name>  AWS (SSO) profile boto3 resolves Bedrock credentials from.
#                         Defaults to the AWS_PROFILE env var (e.g. exported in ~/.zshrc). If a
#                         bedrock= tier is chosen with no profile AND no AWS_ACCESS_KEY_ID, the
#                         install ERRORS. Any bedrock= tier needs boto3 + a valid SSO session:
#                         run 'aws sso login --profile <profile>' first.

DIR="$HOME/model-router"; mkdir -p "$DIR"
ENVF="$DIR/model-router.env"

# Diagnostic logger — always prints to stderr so it never pollutes $() captures.
# Set MODEL_ROUTER_DEBUG=1 for extra verbosity (env dump, python path, etc.).
diag() { printf '[diag] %s\n' "$*" >&2; }

# Self-bootstrap: when run from the skill's scripts/ bundle, copy the whole bundle into
# ~/model-router so the start/stop/uninstall scripts + tag_logger live alongside this one.
# When re-run from ~/model-router (via the alias), this is a no-op.
# When piped via `curl | bash`, BASH_SOURCE[0] is unset (bash never populates it for stdin
# scripts). Guard with :-  and treat an empty result as the curl/stdin case.
_bs="${BASH_SOURCE[0]:-}"
if [ -n "$_bs" ]; then
  SCRIPT_DIR="$(cd "$(dirname "$_bs")" && pwd)"
  diag "script source: file — SCRIPT_DIR=$SCRIPT_DIR"
else
  SCRIPT_DIR=""  # curl/stdin mode — files must be fetched from GitHub
  diag "script source: stdin/curl (BASH_SOURCE unset) — will download from GitHub"
fi
MODELS_CONFIG="${SCRIPT_DIR:+$SCRIPT_DIR/}models.json"

# GitHub repo for fetching additional files when run via curl
GITHUB_REPO="imranq2/coding-model-router"
GITHUB_BRANCH="main"

if [ -z "$SCRIPT_DIR" ] || [ "$SCRIPT_DIR" != "$DIR" ]; then
  # Detect if we're running via curl/stdin: SCRIPT_DIR is empty, or the files don't actually
  # exist there (e.g. SCRIPT_DIR=/dev/fd when using `bash <(cat ...)` process substitution).
  if [ -z "$SCRIPT_DIR" ] || [ ! -f "$SCRIPT_DIR/install-model-router.sh" ]; then
    diag "bootstrap path: curl/stdin → downloading from GitHub ($GITHUB_REPO@$GITHUB_BRANCH)"
    echo "[0/6] Downloading script bundle from GitHub..."
    mkdir -p "$DIR"

    # Download each required file from GitHub
    for file in install-model-router.sh start-model-router.sh stop-model-router.sh uninstall-model-router.sh router.py models.json mcp-local.json; do
      echo "  Fetching $file..."
      if ! curl -fsSL "https://raw.githubusercontent.com/$GITHUB_REPO/$GITHUB_BRANCH/$file" -o "$DIR/$file"; then
        echo "ERROR: Failed to download $file" >&2
        exit 1
      fi
    done

    # Download the wheel if available (GitHub releases or main branch)
    if curl -sfI "https://raw.githubusercontent.com/$GITHUB_REPO/$GITHUB_BRANCH/vllm_mlx-0.4.0-py3-none-any.whl" | head -1 | grep -q "200"; then
      curl -fsSL "https://raw.githubusercontent.com/$GITHUB_REPO/$GITHUB_BRANCH/vllm_mlx-0.4.0-py3-none-any.whl" -o "$DIR/vllm_mlx-0.4.0-py3-none-any.whl"
      echo "  Fetching vllm_mlx-0.4.0-py3-none-any.whl..."
    else
      # Wheel not available on GitHub, will fetch from PyPI during install
      echo "  (vllm_mlx wheel not found on GitHub - will install from PyPI)"
    fi

    chmod +x "$DIR"/*.sh
    # Update SCRIPT_DIR and MODELS_CONFIG to the new location
    SCRIPT_DIR="$DIR"
    MODELS_CONFIG="$DIR/models.json"
  else
    diag "bootstrap path: local bundle → copying from $SCRIPT_DIR"
    echo "[0/6] Copying script bundle to $DIR ..."
    cp "$SCRIPT_DIR/"*.sh "$SCRIPT_DIR/router.py" "$SCRIPT_DIR/models.json" "$SCRIPT_DIR/mcp-local.json" "$DIR/"
    # The .whl is tracked in git but guard against a shallow clone / manual extraction.
    [ -f "$SCRIPT_DIR/vllm_mlx-0.4.0-py3-none-any.whl" ] && \
      cp "$SCRIPT_DIR/vllm_mlx-0.4.0-py3-none-any.whl" "$DIR/"
    chmod +x "$DIR"/*.sh
  fi
else
  diag "bootstrap path: already in $DIR (re-run or alias invocation)"
fi

# Defaults (overridden by a prior install's env, then by flags).
MODE=""
MODEL_ID=""
MODEL_EXPLICIT=""   # set when --model is passed, so it wins over the interactive picker
MLX_PORT=8770
ROUTER_PORT=8771
ANTHROPIC_KEY=""
# Per-tier backend: each of haiku/sonnet/opus/fable is local (mlx-lm) | claude (Anthropic) |
# bedrock (AWS Bedrock). Empty = unset → resolved after flags (default haiku=local, rest claude).
# Set via --haiku/--sonnet/--opus/--fable, or the --local shortcut.
HAIKU_BACKEND="";  SONNET_BACKEND=""; OPUS_BACKEND=""; FABLE_BACKEND=""
HAIKU_BEDROCK="";  SONNET_BEDROCK=""; OPUS_BEDROCK=""; FABLE_BEDROCK=""   # Bedrock id per tier
LOCAL_TIERS=""     # legacy --local / pre-existing env value; folded into the per-tier backends
TIERS_EXPLICIT=""  # set when any tier flag/--local is passed → skip the interactive per-tier menu
MAXTOK_EXPLICIT=""    # set when --max-tokens is passed → skip interactive prompt
PCB_EXPLICIT=""       # set when --prompt-cache-bytes is passed → skip interactive prompt
AUTOCOMPACT_EXPLICIT="" # set when --autocompact-pct is passed → skip interactive prompt
AUTOCOMPACT_ENABLED=0  # 0=off (default), 1=on — controlled by --autocompact / --no-autocompact
AUTOCOMPACT_ENABLED_EXPLICIT=""  # set when either flag is passed → skip interactive prompt
USE_LOCAL_MODELS=""    # 1=yes (default), 0=no — prompt if unset on a TTY, saved in env
MLX_MAX_TOKENS=""          # --max-tokens for vllm-mlx: max output tokens per local response (set below from model_ctx)
PROMPT_CACHE_BYTES=""      # --prompt-cache-bytes passthrough to mlx_lm.server (persisted)
# Context capping for claude-router: tell Claude Code the window is the LOCAL model's, so it
# auto-compacts before a request (esp. background/compaction) overflows the local model — and
# compacts early enough that the KV cache fits RAM. Keeps everything local (no cloud fallback).
AUTO_COMPACT_WINDOW="auto"     # CLAUDE_CODE_AUTO_COMPACT_WINDOW directive: "auto" = the selected
                               # model's context window (via model_ctx); or an explicit token count
AUTOCOMPACT_PCT="55"           # CLAUDE_AUTOCOMPACT_PCT_OVERRIDE — compact at this % (lower = earlier)
# AWS Bedrock auth (used by any tier whose backend is bedrock) — via an AWS SSO profile.
BEDROCK_SONNET_MODEL=""        # legacy env key (pre per-tier flags); migrated to SONNET_BEDROCK
AWS_REGION="us-east-1"         # --aws-region for the Bedrock call
AWS_PROFILE_NAME=""            # --aws-profile: AWS (SSO) profile boto3 resolves creds from

# Reuse prior values on a bare re-run.
# shellcheck source=/dev/null
if [ -f "$ENVF" ]; then
  diag "loading prior env: $ENVF"
  . "$ENVF"
  [ "${MODEL_ROUTER_DEBUG:-0}" = "1" ] && grep -v '^\s*#' "$ENVF" | grep -v '^\s*$' | while IFS= read -r _l; do diag "  env: $_l"; done || true
else
  diag "no prior env found at $ENVF (first install)"
fi

# Tiers are named by COMPLEXITY for the user; Claude Code itself uses model-family names, so we
# map: low→haiku (background/simple), medium→sonnet (daily coding), high→opus (hard reasoning),
# extreme→fable (hardest / long-horizon). ctier accepts a level OR a raw tier (for old-env reuse).
ctier() {
  case "$1" in
    low)                          echo haiku ;;
    medium)                       echo sonnet ;;
    high)                         echo opus ;;
    extreme|extremely-high|xhigh) echo fable ;;
    haiku|sonnet|opus|fable)      echo "$1" ;;
    *)                            echo "" ;;
  esac
}
# Set a tier's backend. $1=level(low|medium|high|extreme) or tier, $2=spec(local|claude|bedrock=<id>).
set_tier_backend() {
  local lvl="$1" val="$2" tier be id=""
  tier="$(ctier "$lvl")"
  [ -n "$tier" ] || { echo "ERROR: unknown tier '$lvl' (use low|medium|high|extreme)" >&2; exit 2; }
  case "$val" in
    local)               be="local" ;;
    claude|anthropic)    be="claude" ;;
    bedrock=*|bedrock:*) be="bedrock"; id="${val#bedrock?}" ;;
    bedrock)             be="bedrock" ;;   # id supplied elsewhere; validated later
    *) echo "ERROR: --$lvl must be 'local', 'claude', or 'bedrock=<model-id>' (got '$val')" >&2; exit 2 ;;
  esac
  case "$tier" in
    haiku)  HAIKU_BACKEND="$be";  [ -n "$id" ] && HAIKU_BEDROCK="$id" ;;
    sonnet) SONNET_BACKEND="$be"; [ -n "$id" ] && SONNET_BEDROCK="$id" ;;
    opus)   OPUS_BACKEND="$be";   [ -n "$id" ] && OPUS_BEDROCK="$id" ;;
    fable)  FABLE_BACKEND="$be";  [ -n "$id" ] && FABLE_BEDROCK="$id" ;;
  esac
  return 0   # the `[ -n "$id" ]` test above is falsy for local/claude — don't leak its exit status (set -e)
}
# --local shortcut: mark levels local. "all"=all local, "none"=all claude. Accepts complexity
# labels (low,medium,high,extreme) or raw tier names (to migrate an old saved LOCAL_TIERS).
apply_local() {
  local v="$1" t; local -a list
  case "$v" in
    all)  for t in low medium high extreme; do set_tier_backend "$t" local;  done; return ;;
    none) for t in low medium high extreme; do set_tier_backend "$t" claude; done; return ;;
  esac
  IFS=',' read -ra list <<< "$v"
  for t in "${list[@]:-}"; do [ -n "$t" ] && set_tier_backend "$t" local; done
  return 0
}

while [ $# -gt 0 ]; do
  case "$1" in
    --mode)         MODE="$2"; shift 2 ;;
    --model)        MODEL_ID="$2"; MODEL_EXPLICIT=1; shift 2 ;;
    --mlx-port)     MLX_PORT="$2"; shift 2 ;;
    --router-port) ROUTER_PORT="$2"; shift 2 ;;
    --api-key)      ANTHROPIC_KEY="$2"; shift 2 ;;
    --low)          set_tier_backend low     "$2"; TIERS_EXPLICIT=1; shift 2 ;;
    --medium)       set_tier_backend medium  "$2"; TIERS_EXPLICIT=1; shift 2 ;;
    --high)         set_tier_backend high    "$2"; TIERS_EXPLICIT=1; shift 2 ;;
    --extreme)      set_tier_backend extreme "$2"; TIERS_EXPLICIT=1; shift 2 ;;
    --local)        apply_local "$2"; TIERS_EXPLICIT=1; shift 2 ;;
    --max-tokens)   MLX_MAX_TOKENS="$2"; MAXTOK_EXPLICIT=1; shift 2 ;;
    --prompt-cache-bytes) PROMPT_CACHE_BYTES="$2"; PCB_EXPLICIT=1; shift 2 ;;
    --auto-compact-window) AUTO_COMPACT_WINDOW="$2"; shift 2 ;;
    --autocompact-pct)     AUTOCOMPACT_PCT="$2"; AUTOCOMPACT_EXPLICIT=1; shift 2 ;;
    --autocompact)         AUTOCOMPACT_ENABLED=1; AUTOCOMPACT_ENABLED_EXPLICIT=1 ;;
    --no-autocompact)      AUTOCOMPACT_ENABLED=0; AUTOCOMPACT_ENABLED_EXPLICIT=1 ;;
    --aws-region)     AWS_REGION="$2"; shift 2 ;;
    --aws-profile)    AWS_PROFILE_NAME="$2"; shift 2 ;;
    *) echo "ERROR: unknown argument: $1" >&2; exit 2 ;;
  esac
done

# ---- Auto-detect anything not supplied via flags or a prior env ----
detect_mode() {  # prints A (subscription), B (api key), or "" (neither)
  if ! command -v python3 >/dev/null 2>&1; then
    printf '[diag] detect_mode: python3 not found — cannot check auth\n' >&2
    echo ""; return
  fi
  python3 - <<'PY'
import json, os, sys

def diag(msg):
    print(f"[diag] {msg}", file=sys.stderr)

oauth = False
try:
    data = json.load(open(os.path.expanduser("~/.claude.json"))) or {}
    oauth = bool(data.get("oauthAccount"))
    if oauth:
        diag("detect_mode: found OAuth account in ~/.claude.json → Mode A")
    else:
        diag("detect_mode: ~/.claude.json present but no oauthAccount key")
except FileNotFoundError:
    diag("detect_mode: ~/.claude.json not found (not logged in to Claude Code)")
except Exception as e:
    diag(f"detect_mode: could not read ~/.claude.json: {e}")

key = bool(os.environ.get("ANTHROPIC_API_KEY"))
if key:
    diag("detect_mode: ANTHROPIC_API_KEY set in environment → Mode B candidate")
if not key:
    try:
        s = json.load(open(os.path.expanduser("~/.claude/settings.json")))
        key = "ANTHROPIC_API_KEY" in (s.get("env") or {})
        if key:
            diag("detect_mode: ANTHROPIC_API_KEY found in ~/.claude/settings.json → Mode B candidate")
        else:
            diag("detect_mode: ~/.claude/settings.json present but no ANTHROPIC_API_KEY in env block")
    except FileNotFoundError:
        diag("detect_mode: ~/.claude/settings.json not found")
    except Exception as e:
        diag(f"detect_mode: could not read ~/.claude/settings.json: {e}")

# Prefer subscription when logged in, so a stray key doesn't flip to paid API billing.
result = "A" if oauth else ("B" if key else "")
if not result:
    diag("detect_mode: no auth found — will error unless --mode A|B passed")
print(result)
PY
}
# ---------------------------------------------------------------------------
# Model config — loaded from models.json once at startup via _init_model_config().
# All per-model data (context windows, sizes, Bedrock ids, tier defaults, RAM
# thresholds) lives in models.json; the functions below are thin variable lookups.
# ---------------------------------------------------------------------------

_model_key() {
  # Convert a model ID string (HuggingFace or Bedrock) to a shell-safe variable suffix.
  # Replaces every non-alphanumeric character with '_'.
  printf '%s' "$1" | sed 's/[^A-Za-z0-9]/_/g'
}

_init_model_config() {
  # Parse models.json with Python (already a prerequisite) and emit shell variable
  # assignments. eval'd once; subsequent lookups read plain shell variables — no subprocess.
  if [ ! -f "$MODELS_CONFIG" ]; then
    echo "WARNING: $MODELS_CONFIG not found — model data will fall back to built-in defaults." >&2
    return
  fi
  local _out
  _out="$(python3 - "$MODELS_CONFIG" <<'PYEOF'
import json, sys, re

def shq(s):
    return "'" + str(s).replace("'", "'\\''") + "'"

def vk(s):
    return re.sub(r'[^A-Za-z0-9]', '_', str(s))

d = json.load(open(sys.argv[1]))

lm = d.get('local_models', [])
print(f"LOCAL_MODEL_COUNT={len(lm)}")
for i, m in enumerate(lm):
    k = vk(m['id'])
    print(f"LOCAL_MODEL_{i}={shq(m['id'])}")
    print(f"LOCAL_SIZE_{i}={shq(str(m.get('size_gb', '')))}")
    print(f"LOCAL_FIT32_{i}={shq(m.get('fit_32gb', '?'))}")
    print(f"LOCAL_FIT16_{i}={shq(m.get('fit_16gb', '?'))}")
    print(f"LOCAL_REASON_{i}={shq(m.get('reason', ''))}")
    print(f"MODEL_CTX_{k}={m.get('context_window', '')}")
    v = m.get('max_output_tokens')
    print(f"MODEL_MAXOUT_{k}={v if v is not None else ''}")
    print(f"MODEL_PARSER_{k}={shq(m.get('tool_parser') or '')}")

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

for tier, val in d.get('defaults', {}).get('tier_backends', {}).items():
    T = tier.upper()
    backend = val.get('backend', 'claude') if isinstance(val, dict) else str(val)
    print(f"DEFAULT_BACKEND_{T}={shq(backend)}")
    if isinstance(val, dict) and 'bedrock_model' in val:
        print(f"DEFAULT_BEDROCK_{T}={shq(val['bedrock_model'])}")
    if isinstance(val, dict) and 'local_model' in val:
        print(f"DEFAULT_LOCAL_MODEL={shq(val['local_model'])}")

rs = d.get('defaults', {}).get('ram_model_selection', [])
print(f"RAM_SEL_COUNT={len(rs)}")
for i, e in enumerate(rs):
    print(f"RAM_SEL_{i}_MIN={e['ram_min_gb']}")
    print(f"RAM_SEL_{i}_MODEL={shq(e['model'])}")

kc = d.get('defaults', {}).get('ram_kv_cache_cap', [])
print(f"KV_CAP_COUNT={len(kc)}")
for i, e in enumerate(kc):
    print(f"KV_CAP_{i}_MAX={e['ram_max_gb']}")
    print(f"KV_CAP_{i}_TOKENS={e['tokens']}")
PYEOF
)"
  eval "$_out"
}

detect_model() {  # choose by installed RAM (GB) using ram_model_selection from models.json
  local gb i min model
  gb=$(( $(sysctl -n hw.memsize 2>/dev/null || echo 0) / 1073741824 ))
  i=0
  while [ "$i" -lt "${RAM_SEL_COUNT:-0}" ]; do
    eval "min=\$RAM_SEL_${i}_MIN; model=\$RAM_SEL_${i}_MODEL"
    if [ "$gb" -ge "$min" ]; then printf '%s' "$model"; return; fi
    i=$(( i + 1 ))
  done
  echo "mlx-community/Qwen3.5-9B-MLX-4bit"  # fallback if models.json not loaded
}

model_ctx() {  # context window (tokens) for a known local model id; empty if unknown
  local k; k="$(_model_key "$1")"
  eval "printf '%s' \"\${MODEL_CTX_${k}:-}\""
}

ram_window_cap() {  # max context (tokens) the local KV cache can safely hold for this much RAM
  # A model's window is an upper bound on POSITIONS, not on what fits: the KV cache grows with
  # context length and is what OOMs the GPU. Cap auto-compaction so the cache stays within RAM.
  local gb i max tokens
  gb=$(( $(sysctl -n hw.memsize 2>/dev/null || echo 0) / 1073741824 ))
  i=0
  while [ "$i" -lt "${KV_CAP_COUNT:-0}" ]; do
    eval "max=\$KV_CAP_${i}_MAX; tokens=\$KV_CAP_${i}_TOKENS"
    if [ "$gb" -le "$max" ]; then printf '%s' "$tokens"; return; fi
    i=$(( i + 1 ))
  done
  echo 32768  # fallback
}

prompt_model() {  # interactive picker. Echoes the chosen HF id on STDOUT; all UI goes to STDERR
  # so the caller can capture it with $(...). $1 = currently-saved model (default to keeping it).
  # Model list, sizes, fit indicators, and reasons all come from models.json.
  local current="$1" rec choice custom i rec_idx="" def_idx gb mark n custom_idx cwin cdisp
  rec="$(detect_model)"
  local -a ids=() sizes=() f32=() f16=() why=()
  i=0
  while [ "$i" -lt "${LOCAL_MODEL_COUNT:-0}" ]; do
    local _id _sz _f32 _f16 _why
    eval "_id=\$LOCAL_MODEL_${i}; _sz=\$LOCAL_SIZE_${i}; _f32=\$LOCAL_FIT32_${i}; _f16=\$LOCAL_FIT16_${i}; _why=\$LOCAL_REASON_${i}"
    ids+=("$_id"); sizes+=("$_sz"); f32+=("$_f32"); f16+=("$_f16"); why+=("$_why")
    i=$(( i + 1 ))
  done
  n=${#ids[@]}; custom_idx=$(( n + 1 ))
  gb=$(( $(sysctl -n hw.memsize 2>/dev/null || echo 0) / 1073741824 ))
  # Default to the RAM auto-pick, unless a prior install saved a model we offer.
  for (( i=1; i<=n; i++ )); do if [ "${ids[$((i-1))]}" = "$rec" ]; then rec_idx=$i; fi; done
  def_idx="${rec_idx:-1}"
  if [ -n "$current" ]; then
    for (( i=1; i<=n; i++ )); do if [ "${ids[$((i-1))]}" = "$current" ]; then def_idx=$i; fi; done
  fi
  {
    echo ""
    echo "Choose the local model (detected RAM: ${gb}GB) — Enter for the auto-pick, or --model <hf-id> to skip:"
    echo ""
    # Emoji render ~2 cells; model/size columns are ASCII-padded; Why column is free text.
    printf "   #  %-34s %-9s %-5s  %-4s %-4s %s\n" "Model" "Size" "Ctx" "32GB" "16GB" "Coding"
    for (( i=1; i<=n; i++ )); do
      mark=""
      if [ "$i" = "$rec_idx" ]; then mark="   <- auto-pick"; fi
      cwin="$(model_ctx "${ids[$((i-1))]}")"
      if [ -n "$cwin" ]; then cdisp="$(( cwin / 1024 ))K"; else cdisp="?"; fi
      printf "  %2d  %-34s %-9s %-5s   %s   %s  %s%s\n" \
        "$i" "${ids[$((i-1))]}" "${sizes[$((i-1))]}" "$cdisp" "${f32[$((i-1))]}" "${f16[$((i-1))]}" "${why[$((i-1))]}" "$mark"
    done
    printf "  %2d  custom — enter any Hugging Face repo id\n" "$custom_idx"
    echo ""
    printf "Selection [%s]: " "$def_idx"
  } >&2
  read -r choice </dev/tty || true
  [ -z "$choice" ] && choice="$def_idx"
  if [ "$choice" = "$custom_idx" ]; then
    printf "Enter Hugging Face repo id: " >&2; read -r custom </dev/tty || true; echo "$custom"
  elif printf '%s' "$choice" | grep -qE '^[0-9]+$' && [ "$choice" -ge 1 ] && [ "$choice" -le "$n" ]; then
    echo "${ids[$((choice-1))]}"
  else
    echo "${ids[$((def_idx-1))]}"   # unrecognized input → the default
  fi
}

# The curated Bedrock model/inference-profile ids — loaded from models.json.
# (region- AND account-specific — confirm what's enabled in your Bedrock console)
bedrock_choices() {
  local i=0
  while [ "$i" -lt "${BEDROCK_MODEL_COUNT:-0}" ]; do
    eval "printf '%s\n' \"\$BEDROCK_MODEL_${i}\""
    i=$(( i + 1 ))
  done
}

bedrock_model_ctx() {  # context window (tokens) for a Bedrock model id; empty if unknown
  local k; k="$(_model_key "$1")"
  eval "printf '%s' \"\${BEDROCK_CTX_${k}:-}\""
}

bedrock_model_max_output() {  # max OUTPUT tokens for a Bedrock model; empty if unknown / not capped
  # Bedrock rejects requests where max_tokens exceeds the model's per-call output ceiling
  # even if the total context window is larger. Claude models are omitted — their 8K limit is correct.
  local k; k="$(_model_key "$1")"
  eval "printf '%s' \"\${BEDROCK_MAXOUT_${k}:-}\""
}

bedrock_model_price() {  # price per million tokens for a Bedrock model id; 0 if unknown
  local k; k="$(_model_key "$1")"
  eval "printf '%s' \"\${BEDROCK_PRICE_${k}:-0}\""
}

tier_price() {  # price per million tokens for an Anthropic tier (haiku/sonnet/opus/fable); 0 if unknown
  local T; T="$(printf '%s' "$1" | tr '[:lower:]' '[:upper:]')"
  eval "printf '%s' \"\${TIER_PRICE_${T}:-0}\""
}

anthropic_model_max_output() {  # max output tokens for an Anthropic model id; empty if unknown
  # Used for passthrough routes so the router raises Claude Code's conservative 8K cap.
  local k; k="$(_model_key "$1")"
  eval "printf '%s' \"\${ANTHROPIC_MAXOUT_${k}:-}\""
}

tier_model_name() {  # the Claude model id for a tier (what "claude" backend routes to)
  local T; T="$(printf '%s' "$1" | tr '[:lower:]' '[:upper:]')"
  eval "printf '%s' \"\${TIER_MODEL_${T}:-}\""
}
prompt_bedrock_model() {  # picker for a Bedrock id; $1 = tier label. Echoes id on STDOUT.
  local label="$1" choice custom i n custom_idx; local -a ids=()
  while IFS= read -r _l; do ids+=("$_l"); done < <(bedrock_choices)
  n=${#ids[@]}; custom_idx=$(( n + 1 ))
  {
    echo ""
    echo "Choose a Bedrock model for the ${label} tier (ids are region/account-specific — confirm in your Bedrock console):"
    for (( i=1; i<=n; i++ )); do
      local _bid="${ids[$((i-1))]}"
      local _p; _p="$(bedrock_model_price "$_bid")"
      local _price_disp="FREE"
      [ "${_p:-0}" != "0" ] && _price_disp="\$${_p}/MTok"
      printf "  %d) %-48s %s\n" "$i" "$_bid" "$_price_disp"
    done
    printf "  %d) custom — enter any Bedrock model id / ARN\n" "$custom_idx"
    printf "Selection [1]: "
  } >&2
  read -r choice </dev/tty || true
  [ -z "$choice" ] && choice=1
  if [ "$choice" = "$custom_idx" ]; then
    printf "Enter Bedrock model id: " >&2; read -r custom </dev/tty || true; echo "$custom"
  elif printf '%s' "$choice" | grep -qE '^[0-9]+$' && [ "$choice" -ge 1 ] && [ "$choice" -le "$n" ]; then
    echo "${ids[$((choice-1))]}"
  else
    echo "${ids[0]}"   # unrecognized → the default (qwen.qwen3-coder-30b-a3b-v1:0)
  fi
}
# Combined per-tier menu: Claude default | the local model | each Bedrock model | custom Bedrock.
# Sets the tier's backend directly. $1=level label, $2=tier(haiku|sonnet|opus|fable).
prompt_tier_backend() {
  local label="$1" tier="$2" choice cid i n def_idx custom_idx; local -a bids=()
  while IFS= read -r _l; do bids+=("$_l"); done < <(bedrock_choices)
  n=${#bids[@]}
  local _tp _tp_disp
  _tp="$(tier_price "$tier")"
  _tp_disp="FREE"; [ "${_tp:-0}" != "0" ] && _tp_disp="\$${_tp}/MTok"

  if [ "${USE_LOCAL_MODELS:-1}" = "1" ]; then
    # Menu: 1=claude, 2=local, 3..(2+n)=bedrock, last=custom
    custom_idx=$(( 2 + n + 1 ))
    case "$(tier_backend "$tier")" in
      local)   def_idx=2 ;;
      bedrock) def_idx=1; for (( i=1; i<=n; i++ )); do [ "${bids[$((i-1))]}" = "$(tier_bedrock "$tier")" ] && def_idx=$(( 2 + i )); done ;;
      *)       def_idx=1 ;;
    esac
    {
      echo ""
      echo "Model for ${label}-complexity tasks ($tier):"
      echo "  1) Claude default — $(tier_model_name "$tier") (Anthropic)  $_tp_disp"
      echo "  2) Local model — ${MODEL_ID}  FREE"
      for (( i=1; i<=n; i++ )); do
        local _bp _bp_disp; _bp="$(bedrock_model_price "${bids[$((i-1))]}")"
        _bp_disp="FREE"; [ "${_bp:-0}" != "0" ] && _bp_disp="\$${_bp}/MTok"
        printf "  %d) Bedrock — %-48s %s\n" "$(( 2 + i ))" "${bids[$((i-1))]}" "$_bp_disp"
      done
      printf "  %d) Bedrock — custom id / ARN\n" "$custom_idx"
      printf "Selection [%s]: " "$def_idx"
    } >&2
    read -r choice </dev/tty || true; [ -z "$choice" ] && choice="$def_idx"
    if   [ "$choice" = 1 ]; then set_tier_backend "$tier" claude
    elif [ "$choice" = 2 ]; then set_tier_backend "$tier" local
    elif [ "$choice" = "$custom_idx" ]; then
      printf "Enter Bedrock model id: " >&2; read -r cid </dev/tty || true
      if [ -n "$cid" ]; then set_tier_backend "$tier" "bedrock=$cid"; else set_tier_backend "$tier" claude; fi
    elif printf '%s' "$choice" | grep -qE '^[0-9]+$' && [ "$choice" -ge 3 ] && [ "$choice" -le "$(( 2 + n ))" ]; then
      set_tier_backend "$tier" "bedrock=${bids[$(( choice - 3 ))]}"
    fi
  else
    # No local models: 1=claude, 2..(1+n)=bedrock, last=custom
    custom_idx=$(( 1 + n + 1 ))
    case "$(tier_backend "$tier")" in
      bedrock) def_idx=1; for (( i=1; i<=n; i++ )); do [ "${bids[$((i-1))]}" = "$(tier_bedrock "$tier")" ] && def_idx=$(( 1 + i )); done ;;
      *)       def_idx=1 ;;
    esac
    {
      echo ""
      echo "Model for ${label}-complexity tasks ($tier):"
      echo "  1) Claude default — $(tier_model_name "$tier") (Anthropic)  $_tp_disp"
      for (( i=1; i<=n; i++ )); do
        local _bp _bp_disp; _bp="$(bedrock_model_price "${bids[$((i-1))]}")"
        _bp_disp="FREE"; [ "${_bp:-0}" != "0" ] && _bp_disp="\$${_bp}/MTok"
        printf "  %d) Bedrock — %-48s %s\n" "$(( 1 + i ))" "${bids[$((i-1))]}" "$_bp_disp"
      done
      printf "  %d) Bedrock — custom id / ARN\n" "$custom_idx"
      printf "Selection [%s]: " "$def_idx"
    } >&2
    read -r choice </dev/tty || true; [ -z "$choice" ] && choice="$def_idx"
    if   [ "$choice" = 1 ]; then set_tier_backend "$tier" claude
    elif [ "$choice" = "$custom_idx" ]; then
      printf "Enter Bedrock model id: " >&2; read -r cid </dev/tty || true
      if [ -n "$cid" ]; then set_tier_backend "$tier" "bedrock=$cid"; else set_tier_backend "$tier" claude; fi
    elif printf '%s' "$choice" | grep -qE '^[0-9]+$' && [ "$choice" -ge 2 ] && [ "$choice" -le "$(( 1 + n ))" ]; then
      set_tier_backend "$tier" "bedrock=${bids[$(( choice - 2 ))]}"
    fi
  fi
  return 0
}
prompt_max_tokens() {  # interactive picker for mlx-lm --max-tokens. Echoes chosen value on STDOUT.
  # $1 = current value (number or "none"); $2 = model context window (tokens).
  local current="${1:-16384}" ctx_window="${2:-0}" choice custom i def_idx
  local -a labels vals
  # Option sets scaled to the model's actual context window.
  if   [ "$ctx_window" -ge 262144 ]; then  # 256K models (Qwen3.5-9B, Qwen3.6-27B, Qwen3-Coder-30B)
    labels=("32 768 — conservative, fastest local responses"
            "65 536 — medium"
            "131 072 — long outputs / large diffs"
            "262 144 — full context window")
    vals=("32768" "65536" "131072" "262144")
  elif [ "$ctx_window" -ge 131072 ]; then  # 128K models (gemma-4)
    labels=("8 192  — conservative, fastest local responses"
            "32 768 — medium"
            "65 536 — long outputs / large diffs"
            "131 072 — full context window")
    vals=("8192" "32768" "65536" "131072")
  else                                      # 40K and smaller models (Qwen3-8B/14B)
    labels=("8 192  — conservative, fastest local responses"
            "16 384 — medium"
            "32 768 — long outputs / large diffs"
            "40 960 — full context window")
    vals=("8192" "16384" "32768" "40960")
  fi
  local n=${#vals[@]} none_idx custom_idx
  none_idx=$(( n + 1 )); custom_idx=$(( n + 2 ))
  def_idx=$none_idx  # default to "don't set"; let vllm-mlx use its built-in default
  [ "$current" = "none" ] && def_idx="$none_idx"
  for (( i=1; i<=n; i++ )); do [ "${vals[$((i-1))]}" = "$current" ] && def_idx=$i; done
  {
    echo ""
    echo "Max output tokens per local-model response (--max-tokens):"
    for (( i=1; i<=n; i++ )); do printf "  %d) %s\n" "$i" "${labels[$((i-1))]}"; done
    printf "  %d) don't set — let vllm-mlx use its built-in default\n" "$none_idx"
    printf "  %d) custom — enter any value\n" "$custom_idx"
    printf "Selection [%d]: " "$def_idx"
  } >&2
  read -r choice </dev/tty || true; [ -z "$choice" ] && choice="$def_idx"
  if [ "$choice" = "$none_idx" ]; then
    echo "none"
  elif [ "$choice" = "$custom_idx" ]; then
    printf "Enter max output tokens: " >&2; read -r custom </dev/tty || true; echo "${custom:-$current}"
  elif printf '%s' "$choice" | grep -qE '^[0-9]+$' && [ "$choice" -ge 1 ] && [ "$choice" -le "$n" ]; then
    echo "${vals[$((choice-1))]}"
  else
    echo "$current"
  fi
}
prompt_autocompact_pct() {  # interactive picker for AUTOCOMPACT_PCT. Echoes chosen value on STDOUT.
  # Lower % = compact earlier = smaller KV cache peak = less OOM risk on RAM-constrained machines.
  local current="${1:-55}" ram_gb choice custom i def_idx
  ram_gb=$(( $(sysctl -n hw.memsize 2>/dev/null || echo 0) / 1073741824 ))
  local -a labels=("35% — compact early (best for ≤16GB RAM, fewest OOM errors)"
                   "45% — moderate (good for 24GB RAM)"
                   "55% — default (comfortable for 32–48GB RAM)"
                   "70% — relaxed (48GB+ RAM, longest context before compaction)")
  local -a vals=("35" "45" "55" "70")
  local n=${#vals[@]} custom_idx; custom_idx=$(( n + 1 )); def_idx=3
  if   [ "$ram_gb" -le 16 ]; then def_idx=1
  elif [ "$ram_gb" -le 24 ]; then def_idx=2; fi
  for (( i=1; i<=n; i++ )); do [ "${vals[$((i-1))]}" = "$current" ] && def_idx=$i; done
  {
    echo ""
    echo "Auto-compact threshold — compact when context reaches X% of the local model's window"
    echo "(lower = compact sooner = smaller KV cache = less OOM risk; detected RAM: ${ram_gb}GB):"
    for (( i=1; i<=n; i++ )); do printf "  %d) %s\n" "$i" "${labels[$((i-1))]}"; done
    printf "  %d) custom — enter 1–100\n" "$custom_idx"
    printf "Selection [%d]: " "$def_idx"
  } >&2
  read -r choice </dev/tty || true; [ -z "$choice" ] && choice="$def_idx"
  if [ "$choice" = "$custom_idx" ]; then
    printf "Enter percentage (1–100): " >&2; read -r custom </dev/tty || true; echo "${custom:-$current}"
  elif printf '%s' "$choice" | grep -qE '^[0-9]+$' && [ "$choice" -ge 1 ] && [ "$choice" -le "$n" ]; then
    echo "${vals[$((choice-1))]}"
  else
    echo "$current"
  fi
}
prompt_autocompact_enabled() {  # yes/no prompt for auto-compact. Echoes 1 (on) or 0 (off) on STDOUT.
  local current="${1:-1}" def_label choice
  def_label="$([ "$current" = "1" ] && echo "Y" || echo "N")"
  {
    echo ""
    echo "Auto-compact — automatically compact the context before hitting the model's limit?"
    echo "(recommended for Bedrock/local models; press Enter to keep current: ${def_label})"
    printf "Enable auto-compact? [y/N]: "
  } >&2
  read -r choice </dev/tty || true
  case "${choice:-$def_label}" in
    [Nn]*) echo 0 ;;
    *)     echo 1 ;;
  esac
}
prompt_prompt_cache() {  # interactive picker for --prompt-cache-bytes. Echoes chosen value on STDOUT.
  local current="${1:-}" choice custom i def_idx
  local -a labels=("disabled — no cross-request KV reuse (most predictable memory)"
                   "2 GB — small cross-request cache"
                   "4 GB — moderate (good balance for 16–24GB RAM)"
                   "8 GB — large (for 32GB+ RAM, faster on repeated prefixes)")
  local -a vals=("" "2147483648" "4294967296" "8589934592")
  local n=${#vals[@]} custom_idx; custom_idx=$(( n + 1 )); def_idx=1
  for (( i=1; i<=n; i++ )); do [ "${vals[$((i-1))]}" = "$current" ] && def_idx=$i; done
  {
    echo ""
    echo "Cross-request KV cache limit (--prompt-cache-bytes):"
    echo "(limits memory used to reuse KV activations across requests; 'disabled' is safest for low RAM):"
    for (( i=1; i<=n; i++ )); do printf "  %d) %s\n" "$i" "${labels[$((i-1))]}"; done
    printf "  %d) custom — enter bytes (e.g. 6GB)\n" "$custom_idx"
    printf "Selection [%d]: " "$def_idx"
  } >&2
  read -r choice </dev/tty || true; [ -z "$choice" ] && choice="$def_idx"
  if [ "$choice" = "$custom_idx" ]; then
    printf "Enter size (e.g. 4294967296 or 4GB): " >&2; read -r custom </dev/tty || true
    # Accept NGB shorthand
    case "${custom:-}" in
      *[Gg][Bb]) echo "$(( ${custom%[Gg][Bb]} * 1073741824 ))" ;;
      *)          echo "${custom:-}" ;;
    esac
  elif printf '%s' "$choice" | grep -qE '^[0-9]+$' && [ "$choice" -ge 1 ] && [ "$choice" -le "$n" ]; then
    echo "${vals[$((choice-1))]}"
  else
    echo "$current"
  fi
}
free_port() {  # first free port at or above $1
  local p="$1" bumped=0
  while lsof -i ":$p" -sTCP:LISTEN >/dev/null 2>&1; do
    diag "port $p busy — bumping to $(( p + 1 ))"
    p=$(( p + 1 )); bumped=1
  done
  [ "$bumped" = "0" ] && diag "port $p free" || diag "port settled at $p"
  echo "$p"
}
# Load all model data from models.json into shell variables (one Python call; fast thereafter).
_init_model_config

[ -n "$MODE" ]     || { MODE="$(detect_mode || true)"; AUTO_MODE=1; }
diag "auth mode: ${MODE:-UNDETECTED}${AUTO_MODE:+ (auto-detected)}${AUTO_MODE:-} "
_ram_gb=$(( $(sysctl -n hw.memsize 2>/dev/null || echo 0) / 1073741824 ))
diag "system RAM: ${_ram_gb}GB"

# Ask whether to use local models (vllm-mlx). Only prompt on a TTY when not already saved.
if [ -t 1 ] && [ -z "${USE_LOCAL_MODELS:-}" ]; then
  printf "\nUse local models (vllm-mlx, requires GPU + model download)? [Y/n]: " >&2
  read -r _ulm </dev/tty || true
  case "${_ulm:-y}" in
    [Nn]*) USE_LOCAL_MODELS=0 ;;
    *)     USE_LOCAL_MODELS=1 ;;
  esac
fi
USE_LOCAL_MODELS="${USE_LOCAL_MODELS:-1}"

# Model selection: an explicit --model always wins. Otherwise, when attached to a terminal,
# prompt the user (defaulting to the saved or RAM-recommended model). With no TTY — the skill
# or launchd invoking us non-interactively — fall back to the saved value or the RAM-based
# recommendation, so we never block waiting on input. Skipped when USE_LOCAL_MODELS=0.
if [ "${USE_LOCAL_MODELS}" = "1" ]; then
  if [ -n "$MODEL_EXPLICIT" ]; then
    :
  elif [ -t 1 ]; then
    [ -n "$MODEL_ID" ] || MODEL_ID="${DEFAULT_LOCAL_MODEL:-$(detect_model)}"
    MODEL_ID="$(prompt_model "$MODEL_ID")"
  else
    [ -n "$MODEL_ID" ] || MODEL_ID="${DEFAULT_LOCAL_MODEL:-$(detect_model)}"
  fi
fi
if [ "$MODE" = "B" ] && [ -z "$ANTHROPIC_KEY" ]; then ANTHROPIC_KEY="${ANTHROPIC_API_KEY:-}"; fi
[ "${USE_LOCAL_MODELS}" = "1" ] && diag "local model selected: $MODEL_ID"

# Interactive local-model prompts (max-tokens, autocompact, prompt-cache) — only needed when
# a local model is actually used.
[ -z "$MLX_MAX_TOKENS" ] && MLX_MAX_TOKENS="none"
if [ "${USE_LOCAL_MODELS}" = "1" ]; then
  _model_ctx_tokens="$(model_ctx "$MODEL_ID")"
  if [ -t 1 ]; then
    [ -z "$MAXTOK_EXPLICIT" ]             && MLX_MAX_TOKENS="$(prompt_max_tokens "$MLX_MAX_TOKENS" "${_model_ctx_tokens:-0}")"
    [ -z "$AUTOCOMPACT_ENABLED_EXPLICIT" ] && AUTOCOMPACT_ENABLED="$(prompt_autocompact_enabled "$AUTOCOMPACT_ENABLED")"
    [ "$AUTOCOMPACT_ENABLED" = "1" ] && [ -z "$AUTOCOMPACT_EXPLICIT" ] && AUTOCOMPACT_PCT="$(prompt_autocompact_pct "$AUTOCOMPACT_PCT")"
    [ -z "$PCB_EXPLICIT" ]               && PROMPT_CACHE_BYTES="$(prompt_prompt_cache "$PROMPT_CACHE_BYTES")"
  fi
fi

# Resolve the auto-compaction window baked into claude-router. "auto" (the default) tracks the
# selected model's actual context window via model_ctx; an explicit number is used verbatim.
# AUTO_COMPACT_WINDOW stays the persisted directive; ACW is the concrete value baked below.
# NOTE: after tier backends are resolved (below) the ACW may be overridden: Bedrock tiers use
# the model's known context window; pure cloud Anthropic tiers use 200K.
ACW="$AUTO_COMPACT_WINDOW"
if [ "${USE_LOCAL_MODELS}" = "1" ] && [ "$ACW" = "auto" ]; then
  _win="$(model_ctx "$MODEL_ID")"; [ -n "$_win" ] || _win=131072   # model window (128K fallback)
  _cap="$(ram_window_cap)"                                          # what this Mac's RAM can hold
  ACW=$(( _win < _cap ? _win : _cap ))                              # min — bound by the tighter limit
  diag "compact window: model=${_win} tokens, ram_cap=${_cap} tokens → ACW=${ACW}"
fi

[ "${USE_LOCAL_MODELS}" = "1" ] && MLX_PORT="$(free_port "$MLX_PORT")"
ROUTER_PORT="$(free_port "$ROUTER_PORT")"
# free_port only sees *listening* sockets, but MLX_PORT was just reserved without binding —
# so the router search can land on the SAME number (both servers then fight for one port and
# Claude Code hits vllm-mlx → 404 on /v1/messages). Force them distinct.
[ "${USE_LOCAL_MODELS}" = "1" ] && [ "$ROUTER_PORT" = "$MLX_PORT" ] && ROUTER_PORT="$(free_port "$(( MLX_PORT + 1 ))")"

# Validate.
[ "$MODE" = "A" ] || [ "$MODE" = "B" ] || { echo "ERROR: could not detect auth mode — log in to Claude Code (subscription) or set ANTHROPIC_API_KEY, or pass --mode A|B." >&2; exit 2; }
[ "${USE_LOCAL_MODELS}" = "1" ] && { [ -n "$MODEL_ID" ] || { echo "ERROR: --model is required" >&2; exit 2; }; }
if [ "$MODE" = "B" ] && [ -z "$ANTHROPIC_KEY" ]; then echo "ERROR: Mode B but no API key — set ANTHROPIC_API_KEY or pass --api-key." >&2; exit 2; fi

# ---- Resolve each tier's backend (local | claude | bedrock) ----
# Accessors over the per-tier vars.
tier_backend() { case "$1" in haiku) echo "$HAIKU_BACKEND";; sonnet) echo "$SONNET_BACKEND";; opus) echo "$OPUS_BACKEND";; fable) echo "$FABLE_BACKEND";; esac; }
tier_bedrock() { case "$1" in haiku) echo "$HAIKU_BEDROCK";; sonnet) echo "$SONNET_BEDROCK";; opus) echo "$OPUS_BEDROCK";; fable) echo "$FABLE_BEDROCK";; esac; }
any_bedrock()  { [ "$HAIKU_BACKEND" = "bedrock" ] || [ "$SONNET_BACKEND" = "bedrock" ] || [ "$OPUS_BACKEND" = "bedrock" ] || [ "$FABLE_BACKEND" = "bedrock" ]; }
any_local()    { [ "$HAIKU_BACKEND" = "local" ] || [ "$SONNET_BACKEND" = "local" ] || [ "$OPUS_BACKEND" = "local" ] || [ "$FABLE_BACKEND" = "local" ]; }

# Migrate a pre-per-tier saved env (LOCAL_TIERS / BEDROCK_SONNET_MODEL) when nothing newer is set.
if [ -z "$HAIKU_BACKEND$SONNET_BACKEND$OPUS_BACKEND$FABLE_BACKEND" ]; then
  [ -n "$LOCAL_TIERS" ] && apply_local "$LOCAL_TIERS"
  [ -n "$BEDROCK_SONNET_MODEL" ] && set_tier_backend medium "bedrock=$BEDROCK_SONNET_MODEL"
fi
# Defaults: low → local (if local enabled) or claude, everything else → claude.
_haiku_default="${USE_LOCAL_MODELS:-1}"; [ "$_haiku_default" = "1" ] && _haiku_default="local" || _haiku_default="claude"
HAIKU_BACKEND="${HAIKU_BACKEND:-${DEFAULT_BACKEND_HAIKU:-$_haiku_default}}"
SONNET_BACKEND="${SONNET_BACKEND:-${DEFAULT_BACKEND_SONNET:-claude}}"
SONNET_BEDROCK="${SONNET_BEDROCK:-${DEFAULT_BEDROCK_SONNET:-}}"
OPUS_BACKEND="${OPUS_BACKEND:-${DEFAULT_BACKEND_OPUS:-claude}}"
FABLE_BACKEND="${FABLE_BACKEND:-${DEFAULT_BACKEND_FABLE:-claude}}"

# Interactive per-tier menu: when on a terminal and no tier flags were passed, walk the user
# through choosing a backend (Claude default / local model / Bedrock) for each complexity tier,
# defaulting to the current/just-resolved value. Skipped non-interactively or when flags set tiers.
if [ -t 1 ] && [ -z "$TIERS_EXPLICIT" ]; then
  prompt_tier_backend low     haiku
  prompt_tier_backend medium  sonnet
  prompt_tier_backend high    opus
  prompt_tier_backend extreme fable
fi

# If AUTO_COMPACT_WINDOW was "auto" AND no main-coding tier (sonnet/opus/fable) is local,
# set ACW from the actual backend context windows rather than the local model's RAM-capped window.
# Bedrock models have their own (smaller) context limits; cloud Anthropic tiers use 200K.
if [ "$AUTO_COMPACT_WINDOW" = "auto" ]; then
  _any_non_haiku_local=0
  _min_bedrock_ctx=0
  for _t in sonnet opus fable; do
    _tbe="$(tier_backend "$_t")"
    if [ "$_tbe" = "local" ]; then
      _any_non_haiku_local=1; break
    elif [ "$_tbe" = "bedrock" ]; then
      _bctx="$(bedrock_model_ctx "$(tier_bedrock "$_t")")"
      [ -z "$_bctx" ] && _bctx=131072  # 128K fallback for unrecognised Bedrock models
      if [ "$_min_bedrock_ctx" = "0" ] || [ "$_bctx" -lt "$_min_bedrock_ctx" ]; then
        _min_bedrock_ctx="$_bctx"
      fi
    fi
  done
  if [ "$_any_non_haiku_local" = "0" ]; then
    if [ "$_min_bedrock_ctx" != "0" ]; then
      ACW="$_min_bedrock_ctx"   # smallest Bedrock tier's window — don't overflow any of them
    else
      ACW=200000                # all non-haiku tiers are cloud Anthropic → 200K
    fi
  fi
fi

# A bedrock tier needs a model id. If none was given, prompt from the list when on a terminal;
# otherwise (skill/launchd) error rather than guess.
for _t in haiku sonnet opus fable; do
  if [ "$(tier_backend "$_t")" = "bedrock" ] && [ -z "$(tier_bedrock "$_t")" ]; then
    if [ -t 1 ]; then
      _bid="$(prompt_bedrock_model "$_t")"
      [ -n "$_bid" ] || { echo "ERROR: no Bedrock model chosen for the $_t tier." >&2; exit 2; }
      set_tier_backend "$_t" "bedrock=$_bid"
    else
      echo "ERROR: the $_t tier is set to bedrock but no model id was given — use 'bedrock=<model-id>'." >&2; exit 2
    fi
  fi
done
# Resolve the AWS profile for any Bedrock tier: --aws-profile flag > saved env > the AWS_PROFILE
# environment variable (e.g. exported in ~/.zshrc). Then REQUIRE auth when Bedrock is in use.
[ -z "$AWS_PROFILE_NAME" ] && AWS_PROFILE_NAME="${AWS_PROFILE:-}"
if any_bedrock && [ -z "$AWS_PROFILE_NAME" ] && [ -z "${AWS_ACCESS_KEY_ID:-}" ]; then
  echo "ERROR: a tier is set to Bedrock but no AWS credentials are configured." >&2
  echo "       Set your profile once and re-run — recommended via ~/.zshrc:" >&2
  echo "         echo 'export AWS_PROFILE=<your-profile>' >> ~/.zshrc && source ~/.zshrc" >&2
  echo "         aws sso login --profile <your-profile>" >&2
  echo "       (or pass --aws-profile <profile>, or export AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY)." >&2
  exit 2
fi

# Per-tier summary line (level=backend), e.g. low=local(Qwen3.5-9B) medium=bedrock(...) high=claude.
_tiersum=""
for _p in "low:haiku" "medium:sonnet" "high:opus" "extreme:fable"; do
  _lvl="${_p%%:*}"; _tr="${_p##*:}"; _be="$(tier_backend "$_tr")"
  case "$_be" in
    local)   _d="local" ;;
    bedrock) _d="bedrock:$(tier_bedrock "$_tr")" ;;
    *)       _d="claude" ;;
  esac
  _tiersum="${_tiersum}  ${_lvl}=${_d}"
done

_pcbnote="off"; [ -n "$PROMPT_CACHE_BYTES" ] && _pcbnote="$(( PROMPT_CACHE_BYTES / 1073741824 ))GB"
_compactnote="off"; [ "$AUTOCOMPACT_ENABLED" = "1" ] && _compactnote="${ACW}@${AUTOCOMPACT_PCT}%"
if [ "${USE_LOCAL_MODELS}" = "1" ]; then
  echo "→ mode=$MODE  model=$MODEL_ID  ports=$MLX_PORT/$ROUTER_PORT  max-tokens=$MLX_MAX_TOKENS  kv-cache=$_pcbnote  compact=${_compactnote}${AUTO_MODE:+  (mode auto-detected; override with --mode A|B)}"
else
  echo "→ mode=$MODE  router-port=$ROUTER_PORT  local-models=off${AUTO_MODE:+  (mode auto-detected; override with --mode A|B)}"
fi
echo "  tiers:${_tiersum}$(any_bedrock && echo "   [bedrock: region $AWS_REGION${AWS_PROFILE_NAME:+, profile $AWS_PROFILE_NAME}]")"

ZRC="$HOME/.zshrc"; touch "$ZRC"
BEGIN="# >>> claude model routing >>>"
END="# <<< claude model routing <<<"

# Packages live in a dedicated venv inside the bundle. start-model-router.sh runs the
# venv's uvicorn and router.py by absolute path, so nothing needs to be on PATH.
VENV="$DIR/venv"

# Seed the venv with Python 3.12 for a consistent toolchain across machines. The default
# python3 can't be trusted: a too-new interpreter (e.g. 3.14) has no cp-wheel for orjson
# (pulled in by mlx-lm), so pip compiles it from Rust source and the build
# hard-stops (PyO3 0.23 maxes out at 3.13). Pin 3.12; fall back to other supported versions
# only if 3.12 is absent, so the installer still works where 3.12 isn't installed.
PY_PREFERRED="3.12"
pick_python() {
  local c
  for c in "python$PY_PREFERRED" python3.13 python3.11 python3.10; do
    command -v "$c" >/dev/null 2>&1 && { command -v "$c"; return; }
  done
  if command -v python3 >/dev/null 2>&1 \
     && python3 -c 'import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,13) else 1)'; then
    command -v python3
  fi
}
PYBIN="$(pick_python)"
if [ -z "$PYBIN" ]; then
  diag "python: none of python${PY_PREFERRED}/3.13/3.11/3.10 found; default python3=$(python3 --version 2>&1 || echo missing)"
  echo "ERROR: need Python $PY_PREFERRED (or 3.10–3.13) to build the venv (default python3 is $(python3 --version 2>&1)). Install it, e.g. 'brew install python@$PY_PREFERRED'." >&2; exit 2
fi
PY_VER="$("$PYBIN" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
diag "python selected: $PYBIN (v$PY_VER, preferred=$PY_PREFERRED)"

# Rebuild the venv whenever it was built on a different Python than the one we picked — this
# heals both a too-new interpreter (a prior 3.14 attempt) and a version drift (e.g. an old
# 3.13 venv when we now pin 3.12), keeping the toolchain consistent.
if [ -x "$VENV/bin/python" ]; then
  EXISTING_VER="$("$VENV/bin/python" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo "?")"
  if [ "$EXISTING_VER" != "$PY_VER" ]; then
    diag "venv: Python mismatch (existing=$EXISTING_VER, target=$PY_VER) — rebuilding"
    echo "  Rebuilding venv: existing Python $EXISTING_VER != target $PY_VER."
    rm -rf "$VENV"
  else
    diag "venv: existing venv matches Python $PY_VER — reusing"
  fi
else
  diag "venv: no existing venv at $VENV — will create"
fi

if [ "${USE_LOCAL_MODELS}" = "1" ]; then
  echo "[1/6] Installing vllm-mlx + FastAPI stack + huggingface_hub (in a dedicated venv)..."
  [ -x "$VENV/bin/python" ] || "$PYBIN" -m venv "$VENV"
  echo "  venv Python: $("$VENV/bin/python" --version 2>&1) (from $PYBIN)"
  "$VENV/bin/python" -m pip install -q --upgrade pip
  _extra_pip=""; any_bedrock && _extra_pip="boto3 openai"
  # shellcheck disable=SC2086
  if [ -f "$SCRIPT_DIR/vllm_mlx-0.4.0-py3-none-any.whl" ]; then
    VLLM_MLX_REPO="$SCRIPT_DIR/vllm_mlx-0.4.0-py3-none-any.whl"
  else
    echo "  (vllm_mlx wheel not found locally — installing from PyPI)"
    VLLM_MLX_REPO="vllm-mlx==0.4.0"
  fi
  "$VENV/bin/python" -m pip install -q \
    "$VLLM_MLX_REPO" \
    "fastapi>=0.111" "uvicorn[standard]>=0.29" "httpx>=0.27" \
    huggingface_hub $_extra_pip
else
  echo "[1/6] Installing FastAPI stack (no local model)..."
  [ -x "$VENV/bin/python" ] || "$PYBIN" -m venv "$VENV"
  echo "  venv Python: $("$VENV/bin/python" --version 2>&1) (from $PYBIN)"
  "$VENV/bin/python" -m pip install -q --upgrade pip
  # shellcheck disable=SC2086
  _extra_pip=""; any_bedrock && _extra_pip="boto3 openai"
  "$VENV/bin/python" -m pip install -q \
    "fastapi>=0.111" "uvicorn[standard]>=0.29" "httpx>=0.27" \
    $_extra_pip
fi

if [ "${USE_LOCAL_MODELS}" = "1" ]; then
  echo "[2/6] Downloading model: $MODEL_ID (cached after first run)..."
  # huggingface_hub renamed its CLI to `hf`; fall back to the old name on older versions.
  if [ -x "$VENV/bin/hf" ]; then HF="$VENV/bin/hf"; else HF="$VENV/bin/huggingface-cli"; fi
  "$HF" download "$MODEL_ID"
else
  echo "[2/6] Skipping model download (local models disabled)."
fi

echo "[3/6] Writing ~/model-router/router_config.json ..."
mkdir -p "$HOME/model-router"

# Build the routes array as JSON. Each tier maps a model name to an upstream type.
# Types: "local" (vllm-mlx), "bedrock" (Bedrock Mantle + SigV4), "anthropic" (direct passthrough).
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
    if printf '%s' "$bid" | grep -qi '^qwen'; then
      printf '    {"tier": "%s", "claude_model": "%s", "url": "https://bedrock-mantle.%s.api.aws/v1/chat/completions", "model": "%s", "auth": "aws", "aws_region": "%s", "api_type": "openai", "price_per_mtok": %s}' \
        "$tier_name" "$claude_model" "$AWS_REGION" "$bid" "$AWS_REGION" "$_bprice"
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

{
  echo "{"
  echo '  "routes": ['
  build_route haiku  "$TIER_MODEL_HAIKU";  echo ","
  build_route sonnet "$TIER_MODEL_SONNET"; echo ","
  build_route opus   "$TIER_MODEL_OPUS";   echo ","
  build_route fable  "$TIER_MODEL_FABLE";  echo ""
  echo "  ]"
  echo "}"
} > "$HOME/model-router/router_config.json"
chmod 600 "$HOME/model-router/router_config.json"
echo "  Written: ~/model-router/router_config.json"

echo "[4/6] Writing $ENVF (sourced by start-model-router.sh + bare re-runs)..."
cat > "$ENVF" <<ENV
# Generated by install-model-router.sh — do not edit by hand; re-run the installer.
MODE="$MODE"
USE_LOCAL_MODELS="$USE_LOCAL_MODELS"
MODEL_ID="$MODEL_ID"
MLX_PORT="$MLX_PORT"
ROUTER_PORT="$ROUTER_PORT"
ANTHROPIC_KEY="$ANTHROPIC_KEY"
HAIKU_BACKEND="$HAIKU_BACKEND"
SONNET_BACKEND="$SONNET_BACKEND"
OPUS_BACKEND="$OPUS_BACKEND"
FABLE_BACKEND="$FABLE_BACKEND"
HAIKU_BEDROCK="$HAIKU_BEDROCK"
SONNET_BEDROCK="$SONNET_BEDROCK"
OPUS_BEDROCK="$OPUS_BEDROCK"
FABLE_BEDROCK="$FABLE_BEDROCK"
MLX_MAX_TOKENS="$MLX_MAX_TOKENS"
PROMPT_CACHE_BYTES="$PROMPT_CACHE_BYTES"
AUTO_COMPACT_WINDOW="$AUTO_COMPACT_WINDOW"
AUTOCOMPACT_ENABLED="$AUTOCOMPACT_ENABLED"
AUTOCOMPACT_PCT="$AUTOCOMPACT_PCT"
AWS_REGION="$AWS_REGION"
AWS_PROFILE_NAME="$AWS_PROFILE_NAME"
ENV
chmod 600 "$ENVF"   # holds keys (Mode B: Anthropic API key)

echo "[5/6] Ensuring bundle scripts are executable..."
chmod +x "$DIR"/*.sh 2>/dev/null || true

echo "[6/6] Adding claude-router + aliases to ~/.zshrc (idempotent)..."
# Remove any prior block first so re-runs don't stack duplicates.
awk 'BEGIN{skip=0}
     /# >>> claude model routing >>>/{skip=1}
     skip==0{print}
     /# <<< claude model routing <<</{skip=0}' "$ZRC" > "$ZRC.tmp" && mv "$ZRC.tmp" "$ZRC"

if [ "$MODE" = "A" ]; then
  cat >> "$ZRC" <<ZBLOCK
$BEGIN
# Run Claude Code WITH local model routing (Mode A — OAuth subscription).
# Plain 'claude' is unaffected and still talks to Anthropic directly.
# Session model is 'opusplan': Opus for plan-mode (complex reasoning), Sonnet for
# execution (normal coding); Haiku still handles background tasks. Override per run
# with 'claude-router --model sonnet' or /model.
claude-router() {
  echo "[model-router] proxy=localhost:$ROUTER_PORT" >&2
  env -u ANTHROPIC_API_KEY \\
    ANTHROPIC_BASE_URL="http://localhost:$ROUTER_PORT" \\
    ANTHROPIC_MODEL="opusplan" \\
    claude \\
      "\$@"
}
alias install-model-router="bash \$HOME/model-router/install-model-router.sh"
alias uninstall-model-router="bash \$HOME/model-router/uninstall-model-router.sh"
alias start-model-router="bash \$HOME/model-router/start-model-router.sh"
alias stop-model-router="bash \$HOME/model-router/stop-model-router.sh"
$END
ZBLOCK
else
  cat >> "$ZRC" <<ZBLOCK
$BEGIN
# Run Claude Code WITH local model routing (Mode B — API key).
# Plain 'claude' is unaffected and still talks to Anthropic directly.
# Session model is 'opusplan': Opus for plan-mode (complex reasoning), Sonnet for
# execution (normal coding); Haiku still handles background tasks. Override per run
# with 'claude-router --model sonnet' or /model.
claude-router() {
  echo "[model-router] proxy=localhost:$ROUTER_PORT" >&2
  ANTHROPIC_BASE_URL="http://localhost:$ROUTER_PORT" \\
    ANTHROPIC_API_KEY="$ANTHROPIC_KEY" \\
    ANTHROPIC_MODEL="opusplan" \\
    claude \\
      "\$@"
}
alias install-model-router="bash \$HOME/model-router/install-model-router.sh"
alias uninstall-model-router="bash \$HOME/model-router/uninstall-model-router.sh"
alias start-model-router="bash \$HOME/model-router/start-model-router.sh"
alias stop-model-router="bash \$HOME/model-router/stop-model-router.sh"
$END
ZBLOCK
fi

echo ""
echo "✓ Install complete. Bundle lives in ~/model-router/"
echo "  This step does NOT start the servers."
echo "  Start the stack:  start-model-router"
echo "  Then OPEN A NEW TERMINAL and run:  claude-router"
echo "  Stop the stack:   stop-model-router"
echo "  Plain 'claude' still uses Anthropic directly."
