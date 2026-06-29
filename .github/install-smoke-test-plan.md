# Plan: GitHub Actions macOS smoke test for `install-model-router.sh`

## Goal

Catch regressions in the install script on every PR — specifically the bootstrap path,
Python/venv setup, file generation, and `.zshrc` patching — without downloading multi-GB
models in CI.

---

## Runner choice

Use `macos-15` (Apple Silicon, M1) as the primary runner — it's what most users of this
project have, and mlx/vllm-mlx only works on Apple Silicon anyway. Add `macos-26` as a
second matrix entry; the linked announcement currently shows it as Intel-only, so it's
useful for validating broad macOS compatibility but cannot test the local-model path.

```yaml
strategy:
  matrix:
    os: [macos-15, macos-26]
```

---

## Two test scenarios per job

The install script has two meaningfully different code paths. Both need coverage:

| Scenario | How | What it exercises |
|---|---|---|
| **A — local file** | `bash install-model-router.sh --mode B` | Normal `SCRIPT_DIR` detection, local bundle copy |
| **B — stdin sim** | `bash <(cat install-model-router.sh) --mode B` | The `BASH_SOURCE` unbound fix — same code path as `curl \| bash` but using the PR's local copy |

Scenario B is the regression test for the `BASH_SOURCE[0]: unbound variable` bug. Using
`<(cat ...)` instead of a real `curl` pipe means the PR's version is tested rather than
whatever is on `main`.

---

## Skipping model download

Pass `USE_LOCAL_MODELS=0` as an env var. The script handles this gracefully (skips the
vllm-mlx install and model download steps, still installs the FastAPI stack and writes all
config). This keeps CI under ~60 seconds instead of 10+ minutes.

A separate optional job (manual trigger or schedule only, not on PR) can be added later to
test the full local-model path with a small model.

---

## Auth

The install script never calls the Anthropic API during install — it only validates that a
key is present. A placeholder value (`sk-ant-ci-placeholder`) is sufficient for install
verification. Use a real key from `secrets.ANTHROPIC_API_KEY` only if a router startup
test is added later.

---

## Assertions after install

After each scenario, assert the outputs that must exist:

```bash
# Bundle files were written
test -f ~/model-router/router_config.json
test -f ~/model-router/model-router.env
test -f ~/model-router/install-model-router.sh
test -f ~/model-router/router.py
test -f ~/model-router/mcp-local.json

# router_config.json is valid JSON with exactly 4 routes
python3 -c "
import json
r = json.load(open('$HOME/model-router/router_config.json'))
assert len(r['routes']) == 4, f'expected 4 routes, got {len(r[\"routes\"])}'
"

# .zshrc block was written
grep -q 'claude-router'       ~/.zshrc
grep -q 'claude model routing' ~/.zshrc

# venv exists with required packages
test -x ~/model-router/venv/bin/python
~/model-router/venv/bin/python -c "import fastapi, uvicorn, httpx"
```

---

## Trigger and path filter

Run on PR only when relevant files change. Skip on docs-only changes:

```yaml
on:
  pull_request:
    paths:
      - 'install-model-router.sh'
      - 'start-model-router.sh'
      - 'stop-model-router.sh'
      - 'uninstall-model-router.sh'
      - 'router.py'
      - 'models.json'
      - 'mcp-local.json'
```

---

## Caching

Cache the pip download cache (not the venv itself — the venv rebuild test is important).
The venv is fast to recreate once wheels are cached:

```yaml
- uses: actions/cache@v4
  with:
    path: ~/Library/Caches/pip
    key: pip-${{ runner.os }}-${{ hashFiles('install-model-router.sh') }}
```

---

## What this does NOT cover (explicit non-goals for now)

- **Model download / MLX inference** — too slow and large for PR CI; can be added as a
  separate scheduled workflow later.
- **Bedrock routes** — requires AWS SSO session; not suitable for PR CI.
- **`claude-router` actually starting** — requires a live API key and running servers;
  out of scope for the install smoke test.
- **`macos-26` Apple Silicon** — not yet available; revisit when confirmed.

---

## File to create

`.github/workflows/install-smoke-test.yml`
