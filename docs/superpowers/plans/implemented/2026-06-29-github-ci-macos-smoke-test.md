# GitHub Actions macOS Smoke Test — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement
> this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Catch regressions in `install-model-router.sh` on every PR — specifically the
bootstrap path, Python/venv setup, file generation, and `.zshrc` patching — without
downloading multi-GB models in CI.

**Architecture:** A single GitHub Actions workflow file with a 2×2 matrix (runner ×
scenario). Both runners are arm64 M1 (`macos-15`, `macos-26`). Two test scenarios per job:
(A) local file execution — tests normal `SCRIPT_DIR` detection and bundle copy; (B) stdin
simulation via `bash <(cat ...)` — tests the `BASH_SOURCE[0]` unbound-variable fix (the
same code path as `curl | bash`, but using the PR's local copy). `USE_LOCAL_MODELS=0`
skips the multi-GB model download so the job completes in under 60 seconds.

**Tech stack:** GitHub Actions, macOS 15 + macOS 26 (arm64 M1, 3 vCPU 7 GB), bash,
Python 3.12, pip cache via `actions/cache`.

## Global Constraints

- **No real API calls during install.** The install script never calls the Anthropic API;
  a placeholder key (`sk-ant-ci-placeholder`) is enough to pass the Mode B key-presence
  check. No `secrets.ANTHROPIC_API_KEY` is required.
- **`USE_LOCAL_MODELS=0` in all CI jobs.** Skips vllm-mlx install and model download.
  The FastAPI stack (fastapi, uvicorn, httpx) is still installed and tested.
- **Intel runners (`macos-15-intel`, `macos-26-intel`) are excluded.** vllm-mlx / MLX
  only works on Apple Silicon; Intel runners offer no additional install-path coverage.
- **No `--no-verify` commits.** This repo has no pre-commit hooks configured; standard
  git commit is fine.
- **Path filter on PR trigger.** The workflow only runs when install-relevant files change;
  docs-only PRs do not trigger it.

---

## Phase 1 — Create the workflow file

### Task 1: Create `.github/workflows/install-smoke-test.yml`

**Files:**
- Create: `.github/workflows/install-smoke-test.yml`

**Interfaces:**
- Produces: a GitHub Actions workflow triggered on PR for the listed paths.
- Consumes: nothing external (no secrets, no services).

- [ ] **Step 1: Create the `.github/workflows/` directory if absent**

```bash
mkdir -p .github/workflows
```

- [ ] **Step 2: Write the workflow file**

Create `.github/workflows/install-smoke-test.yml`:

```yaml
name: Install smoke test

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

jobs:
  smoke-test:
    name: ${{ matrix.os }} / ${{ matrix.scenario }}
    runs-on: ${{ matrix.os }}
    strategy:
      fail-fast: false
      matrix:
        os: [macos-15, macos-26]
        scenario: [local-file, stdin-sim]

    env:
      ANTHROPIC_API_KEY: sk-ant-ci-placeholder
      USE_LOCAL_MODELS: "0"

    steps:
      - uses: actions/checkout@v4

      - name: Cache pip downloads
        uses: actions/cache@v4
        with:
          path: ~/Library/Caches/pip
          key: pip-${{ runner.os }}-${{ hashFiles('install-model-router.sh') }}
          restore-keys: pip-${{ runner.os }}-

      - name: Run install (local file)
        if: matrix.scenario == 'local-file'
        run: bash install-model-router.sh --mode B

      - name: Run install (stdin simulation — mirrors curl | bash)
        if: matrix.scenario == 'stdin-sim'
        run: bash <(cat install-model-router.sh) --mode B

      - name: Verify bundle files exist
        run: |
          test -f ~/model-router/router_config.json
          test -f ~/model-router/model-router.env
          test -f ~/model-router/install-model-router.sh
          test -f ~/model-router/router.py
          test -f ~/model-router/mcp-local.json

      - name: Verify router_config.json has 4 routes
        run: |
          python3 - <<'PY'
          import json, sys
          r = json.load(open(__import__('os').path.expanduser('~/model-router/router_config.json')))
          n = len(r.get('routes', []))
          assert n == 4, f'expected 4 routes, got {n}'
          print(f'OK: {n} routes found')
          PY

      - name: Verify .zshrc block was written
        run: |
          grep -q 'claude-router'        ~/.zshrc
          grep -q 'claude model routing' ~/.zshrc
          echo "OK: .zshrc block present"

      - name: Verify venv and packages
        run: |
          test -x ~/model-router/venv/bin/python
          ~/model-router/venv/bin/python -c "import fastapi, uvicorn, httpx; print('OK: packages present')"
```

- [ ] **Step 3: Remove the old ad-hoc plan file from `.github/`**

```bash
git rm .github/install-smoke-test-plan.md
```

- [ ] **Step 4: Verify the workflow YAML parses cleanly**

```bash
python3 -c "import yaml; yaml.safe_load(open('.github/workflows/install-smoke-test.yml')); print('YAML OK')"
```

Expected: `YAML OK`. If PyYAML is not installed, run `pip install pyyaml` first.

- [ ] **Step 5: Commit**

```bash
git add .github/workflows/install-smoke-test.yml
git commit -m "ci: add macOS arm64 smoke test for install-model-router.sh"
```

---

## Phase 2 — Documentation and index

### Task 2: Remove the old plan file and keep the docs clean

**Files:**
- Delete: `.github/install-smoke-test-plan.md` (moved into `docs/superpowers/plans/`)
- Verify: `docs/superpowers/README.md` index entry matches the plan filename

- [ ] **Step 1: Confirm the old plan file is removed**

```bash
ls .github/install-smoke-test-plan.md 2>/dev/null && echo "STILL EXISTS — remove it" || echo "OK: already gone"
```

- [ ] **Step 2: Confirm the README index is accurate**

Check that `docs/superpowers/README.md` lists this plan under `## 👉 Active`.

- [ ] **Step 3: Commit docs**

```bash
git add docs/superpowers/
git commit -m "docs: add superpowers plan index and CI smoke test plan"
```

---

## Final verification

- [ ] **Push the branch and open a PR** to confirm the workflow appears in the Actions tab
  and all 4 matrix jobs are listed (macos-15/local-file, macos-15/stdin-sim,
  macos-26/local-file, macos-26/stdin-sim).

- [ ] **Confirm all 4 jobs pass** (green checks). Expected runtime: under 3 minutes per
  job (venv creation + FastAPI package install dominates).

- [ ] **Confirm the stdin-sim job passes** specifically — this is the regression test for
  the `BASH_SOURCE[0]: unbound variable` bug fixed in the same PR cycle.

---

## Self-Review

**Scenario coverage:**
- Local-file path (normal `SCRIPT_DIR` detection + bundle copy) → `local-file` scenario. ✅
- stdin/curl path (`BASH_SOURCE` unbound fix) → `stdin-sim` scenario via `bash <(cat ...)`. ✅
- Both arm64 M1 runners (macOS 15 and macOS 26) → matrix. ✅
- Intel runners excluded (MLX-incompatible, no added coverage for this project). ✅

**What is NOT tested (intentional non-goals):**
- Model download / MLX inference — too slow for PR CI; add as a scheduled workflow later.
- Bedrock routes — requires AWS SSO; not suitable for PR CI.
- `claude-router` actually starting — needs a live API key + running servers.

**Auth approach:** `sk-ant-ci-placeholder` as `ANTHROPIC_API_KEY` is sufficient because
the install script only checks for the key's _presence_ (Mode B detection); it never
makes an Anthropic API call. No GitHub secret is required.

**Deviation from draft plan:** the draft discussed a `macos-26` Intel concern — confirmed
from the runner spec table that both `macos-15` and `macos-26` labels resolve to arm64 M1
(3 vCPU, 7 GB). The Intel variants (`macos-15-intel`, `macos-26-intel`) are explicitly
excluded.
