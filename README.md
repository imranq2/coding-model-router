# coding-model-router

A routing proxy for Claude Code that directs each model tier to a different backend — local GPU, AWS Bedrock, or Anthropic directly. Local GPU support is optional; you can route entirely through cloud backends.

## What This Does

Claude Code sends every request to a single Anthropic Messages API endpoint. This proxy intercepts those requests and re-routes them based on the model name in the request body:

```
┌─────────────────────────────────────────────────────────────────────┐
│  router.py (:8771) - FastAPI proxy                                  │
│  Routes by 'model' field in request body:                           │
│    • claude-haiku-4-5-20251001 → vllm-mlx (:8770) [local GPU]  *   │
│    • claude-sonnet-4-6         → Bedrock Mantle [AWS]           *   │
│    • claude-opus-4-8           → Anthropic API [cloud]              │
│    • claude-fable-5            → Anthropic API [cloud]              │
│                                                                     │
│  * each tier's backend is configurable; all tiers can be cloud      │
└─────────────────────────────────────────────────────────────────────┘
```

### `claude` vs `claude-router`

Routing is **strictly opt-in**. The normal `claude` command is completely unaffected — it bypasses the proxy and talks to Anthropic directly.

```
  claude (unchanged)              claude-router (opt-in)
  ──────────────────              ──────────────────────
  No env vars changed             Sets for this invocation only:
                                    ANTHROPIC_BASE_URL=http://localhost:8771

         │                                    │
         ▼                                    ▼
  ┌─────────────┐                  ┌─────────────────────┐
  │ Anthropic   │                  │ router.py (:8771)   │
  │ API         │                  │ proxy               │
  │ (direct)    │                  └──────────┬──────────┘
  └─────────────┘                             │
                               ┌──────────────┼──────────────┐
                               ▼              ▼              ▼
                          ┌────────┐   ┌──────────┐   ┌──────────┐
                          │ vllm-  │   │  Bedrock │   │Anthropic │
                          │ mlx    │   │  Mantle  │   │  API     │
                          │ (:8770)│   │          │   │          │
                          └────────┘   └──────────┘   └──────────┘
                          (optional)   (optional)      (always avail)
```

`claude-router` sets `ANTHROPIC_BASE_URL` only for its own invocation — it does not persist, does not affect other terminals, and does not modify any global config.

### Key Features

- **Opt-in routing**: Plain `claude` talks to Anthropic directly; use `claude-router` to enable routing
- **Tiered routing**: Each Claude tier (Haiku/Sonnet/Opus/Fable) can route to any backend independently
- **Local GPU inference**: Optional — uses vllm-mlx for Apple Silicon acceleration when enabled
- **No API key required**: Works with both OAuth subscription (Mode A) and API key (Mode B)
- **Cost tracking**: Shows token usage and savings in real-time
- **Tool loop support**: Full tool calling works end-to-end

### Use Cases

- Reduce API costs on pay-per-token plans by routing cheaper tiers to Bedrock or local
- Run background tasks (compaction, summarization) locally for privacy
- Offline coding with limited cloud access (local GPU only)
- Test open-weight models transparently inside Claude Code

## Prerequisites

### Required

- **macOS** (Intel or Apple Silicon)
- **Python 3.10–3.13** — installer prefers 3.12:
  ```bash
  brew install python@3.12
  ```
- **Homebrew** (recommended):
  ```bash
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  ```
- **Xcode Command Line Tools** — to compile native Python packages:
  ```bash
  xcode-select --install
  ```

### Optional: AWS Bedrock routing

Required only if you want to route any tier through AWS Bedrock:

```bash
brew install awscli
aws configure   # or: aws sso login --profile <profile>
```

### Optional: Local GPU inference (Apple Silicon only)

Required only if you want to run a local model on-device:

- **Apple Silicon Mac** — M1, M2, M3, or M4 (any variant)
- **RAM**: 16GB minimum; 24GB+ recommended for larger models
- **Storage**: 15–30GB free for model downloads
- **macOS 12 Monterey or later** — required for Metal GPU acceleration

## Installation

### Option 1: Direct from GitHub (Recommended)

```bash
curl -fsSL https://raw.githubusercontent.com/imranq2/coding-model-router/main/install-model-router.sh | bash
```

The installer fetches all required files from GitHub and sets everything up in `~/model-router/`.

### Option 2: From a local clone

```bash
git clone https://github.com/imranq2/coding-model-router.git
cd coding-model-router
bash install-model-router.sh
```

The installer will:
1. Detect your auth mode (OAuth or API key)
2. Ask which backend to use for each tier (local / Bedrock / Anthropic)
3. If local GPU: pick a model based on your Mac's RAM
4. Install dependencies to `~/model-router/venv`
5. Add shell aliases to `~/.zshrc`

## Commands

| Command | Description |
|---------|-------------|
| `claude-router [args]` | Run Claude Code with routing enabled |
| `start-model-router` | Start router (and vllm-mlx if local GPU enabled) |
| `stop-model-router` | Stop the routing stack |
| `install-model-router` | Re-run installer (preserves existing config) |
| `uninstall-model-router` | Full teardown |

## Files

| File | Purpose |
|---|---|
| `install-model-router.sh` | Interactive installer |
| `start-model-router.sh` | Starts router.py (and vllm-mlx if configured) |
| `stop-model-router.sh` | Kills server processes |
| `uninstall-model-router.sh` | Removes config and scripts |
| `router.py` | FastAPI proxy with Anthropic/OpenAI translation |
| `models.json` | Model config (context windows, sizes, Bedrock IDs) |
| `mcp-local.json` | MCP server configuration |
