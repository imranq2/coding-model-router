# coding-model-router

Local GPU-accelerated model routing for Claude Code on Apple Silicon Macs (M1/M2/M3/M4).

## What This Code Does

This project provides a routing proxy that lets Claude Code use local GPU models for background/cheap tasks while keeping complex reasoning on cloud APIs (Anthropic or AWS Bedrock).

### How It Works

```
┌─────────────────────────────────────────────────────────────────────┐
│  claude-router (shell function)                                     │
│  Sets env vars for ONE invocation only                              │
│    • ANTHROPIC_BASE_URL=http://localhost:8771                       │
│    • ANTHROPIC_MODEL=opusplan (Opus for planning, Sonnet for exec)  │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  router.py (:8771) - FastAPI proxy                                  │
│  Routes by 'model' field in request body:                           │
│    • claude-haiku-4-5-20251001 → vllm-mlx (:8770) [local GPU]       │
│    • claude-sonnet-4-6         → Bedrock Mantle [AWS]              │
│    • claude-opus-4-8           → Anthropic API [cloud]              │
│    • claude-fable-5            → Anthropic API [cloud]              │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
               ┌───────────────┼───────────────┐
               ▼               ▼               ▼
          ┌────────┐    ┌──────────┐    ┌──────────┐
          │ vllm-  │    │  Bedrock │    │ Anthropic│
          │ mlx    │    │  Mantle  │    │  API     │
          │ (:8770)│    │          │    │          │
          └────────┘    └──────────┘    └──────────┘
          Local model   AWS GPU       Cloud
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
|------|---------|
| `install-model-router.sh` | Interactive installer script |
| `start-model-router.sh` | Starts vllm-mlx + router.py |
| `stop-model-router.sh` | Kills both server processes |
| `uninstall-model-router.sh` | Removes config and scripts |
| `router.py` | FastAPI proxy with Anthropic/OpenAI translation |
| `models.json` | Model config (context windows, sizes, Bedrock IDs) |
| `mcp-local.json` | MCP server configuration for local tools |
| `vllm_mlx-0.4.0-py3-none-any.whl` | vllm-mlx wheel (bundled for offline install) |

## Original Location

These scripts were originally in `plugins/software-developers/skills/install-model-router/scripts/` in the [bwell-ai-plugin-marketplace](https://github.com/icanbwell/bwell-ai-plugin-marketplace) repository.
