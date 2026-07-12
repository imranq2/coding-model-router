# Plans — index (START HERE)

Implementation plans produced via the brainstorm → plan → implement workflow.
Each plan lives under `plans/` with a date-stamped name.

**The folders are the source of truth** — this index is a map, not an exhaustive list.

## Folder convention

`plans/` is split by status:

- **`active/`** — current, in-flight work; what to build from now. Keep this small.
- **`implemented/`** — shipped to `main`; kept as reference for how/why things work.
- **`superseded/`** — historical, replaced by a later design; do not follow.

When a plan ships, `git mv` it into `implemented/`. When a plan is replaced, `git mv` it
to `superseded/` and add a pointer banner at the top.

## 👉 Active (in flight)

- **[`plans/active/2026-07-12-port-language-model-gateway-improvements.md`](plans/active/2026-07-12-port-language-model-gateway-improvements.md)**
  — Split `router.py` into modules mirroring `language-model-gateway`'s layout, port 4 bug fixes + tokenizer-based context budgeting + regex route fallback. All 12 tasks implemented and reviewed on branch `port-language-model-gateway-improvements`; move to `implemented/` once merged to `main`.

## Implemented (shipped — see `plans/implemented/`)

- **[`plans/implemented/2026-06-29-github-ci-macos-smoke-test.md`](plans/implemented/2026-06-29-github-ci-macos-smoke-test.md)**
  — GitHub Actions macOS smoke test for `install-model-router.sh` on every PR.

## Superseded (historical — see `plans/superseded/`)

_(none yet)_
