# coding-model-router

Local GPU-accelerated model routing for Claude Code on Apple Silicon Macs (M1/M2/M3/M4).

## Prerequisites

> **macOS only.** This project uses vllm-mlx for Apple Silicon GPU inference and will not run on Linux, Windows, or Intel Macs.

### Supported Hardware

- **Mac**: Apple Silicon — M1, M2, M3, or M4 (any variant: Pro, Max, Ultra)
- **RAM**: 16GB minimum; 24GB+ recommended for larger models (the model must fit in unified memory)
- **Storage**: 15–30GB free space for model downloads (varies by model)

### macOS Version

- **macOS 12 Monterey or later** — required for Metal GPU acceleration used by vllm-mlx

### Required macOS Setup

1. **Xcode Command Line Tools** — needed to compile native Python packages:
   ```bash
   xcode-select --install
   ```
   If already installed you'll see `"xcode-select: error: command line tools are already installed"` — that's fine.

2. **Homebrew** — recommended package manager for macOS:
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

3. **Python 3.10–3.13** — the installer prefers Python 3.12:
   ```bash
   brew install python@3.12
   ```

4. **AWS CLI** *(optional)* — only needed if routing Sonnet through AWS Bedrock:
   ```bash
   brew install awscli
   aws configure   # enter your AWS credentials
   ```

### Network

- Initial install requires internet access to download Python packages and models
- Once installed, local models work fully offline; cloud backends (Anthropic/Bedrock) still require internet

## What This Code Does

This project provides a routing proxy that lets Claude Code use local GPU models for background/cheap tasks while keeping complex reasoning on cloud APIs (Anthropic or AWS Bedrock).

### `claude` vs `claude-router`

Routing is **strictly opt-in**. The normal `claude` command is completely unaffected — it bypasses the proxy entirely and talks to Anthropic directly, exactly as it did before installation.

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
                          Haiku→local  Sonnet→AWS    Opus/Fable→cloud
```

`claude-router` sets `ANTHROPIC_BASE_URL` only for its own shell function invocation — it does not persist, does not affect other terminals, and does not modify any global config. As soon as that session ends, the env var is gone.

### How the Router Works

```
┌─────────────────────────────────────────────────────────────────────┐
│  router.py (:8771) - FastAPI proxy                                  │
│  Routes by 'model' field in request body:                           │
│    • claude-haiku-4-5-20251001 → vllm-mlx (:8770) [local GPU]      │
│    • claude-sonnet-4-6         → Bedrock Mantle [AWS]               │
│    • claude-opus-4-8           → Anthropic API [cloud]              │
│    • claude-fable-5            → Anthropic API [cloud]              │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Features

- **Opt-in routing**: Plain `claude` talks to Anthropic directly; use `claude-router` to enable routing
- **Tiered routing**: Each Claude tier (Haiku/Sonnet/Opus/Fable) can route to a different backend
- **Local GPU inference**: Uses vllm-mlx for native Apple Silicon acceleration
- **No API key required**: Works with both OAuth subscription (Mode A) and API key (Mode B)
- **Cost tracking**: Shows token usage and savings in real-time
- **Tool loop support**: Full tool calling works end-to-end

### Use Cases

- Reduce API costs on pay-per-token plans
- Run background tasks (compaction, summarization) locally for privacy
- Offline coding with limited cloud access
- Test models before moving to cloud

## Installation

### Option 1: Direct from GitHub (Recommended)

```bash
curl -fsSL https://raw.githubusercontent.com/imranq2/coding-model-router/main/install-model-router.sh | bash
```

The installer will automatically fetch all required files (router.py, models.json, etc.) from GitHub and set everything up in `~/model-router/`.

### Option 2: Local

```bash
cd /Users/imranqureshi/git/coding-model-router
bash install-model-router.sh
```

The installer will:
1. Detect your auth mode (OAuth or API key)
2. Pick a local model based on your Mac's RAM
3. Walk you through per-tier backend choices
4. Install dependencies to `~/model-router/venv`
5. Add shell aliases to `~/.zshrc`

## Commands

| Command | Description |
|---------|-------------|
| `claude-router [args]` | Run Claude Code with routing enabled |
| `start-model-router` | Start vllm-mlx + router (long-running) |
| `stop-model-router` | Stop the routing stack |
| `install-model-router` | Re-run installer (preserves config) |
| `uninstall-model-router` | Full teardown |

## Files

| File | Purpose |
|---|---|
| `install-model-router.sh` | Interactive installer script |
| `start-model-router.sh` | Starts vllm-mlx + router.py |
| `stop-model-router.sh` | Kills both server processes |
| `uninstall-model-router.sh` | Removes config and scripts |
| `router.py` | FastAPI proxy with Anthropic/OpenAI translation |
| `models.json` | Model config (context windows, sizes, Bedrock IDs) |
| `mcp-local.json` | MCP server configuration for local tools |
| `vllm_mlx-0.4.0-py3-none-any.whl` | vllm-mlx wheel (bundled for offline install, or fetched from PyPI) |

