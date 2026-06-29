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

_(none)_

## Implemented (shipped — see `plans/implemented/`)

- **[`plans/implemented/2026-06-29-github-ci-macos-smoke-test.md`](plans/implemented/2026-06-29-github-ci-macos-smoke-test.md)**
  — GitHub Actions macOS smoke test for `install-model-router.sh` on every PR.

## Superseded (historical — see `plans/superseded/`)

_(none yet)_
