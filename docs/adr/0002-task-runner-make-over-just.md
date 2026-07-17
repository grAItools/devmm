# 2. Task runner: `make` over `just`

## Status

Accepted

## Context

The devmm implementation plan
([`specs/devmm-implementation-plan.md`](../../specs/devmm-implementation-plan.md),
§0) fixes the repository tooling and names **`just`** as the command runner, with
`justfile` targets `just lint`, `just typecheck`, `just test`, `just gate N`, and
`just gate-all` driving the phased build.

This repository was already initialized from the `harness-copier-template` with
**`make`** as the task runner (`task_runner=make`; see `.copier-answers.yml`).
`make` is not incidental here — it is load-bearing across the harness:

- The Claude Code `Stop` hook and the `/verify` slash command invoke
  `make verify`.
- OpenCode's config and the four-phase agent loop reference the same `make`
  targets.
- `make` is already installed on the dev machines and CI images; `just` is not
  (`command -v just` fails locally), so adopting it adds an install step to every
  environment and every CI leg.

Running two task runners (a `justfile` for the build plan and a `Makefile` for
the harness gate) would split the source of truth: the `make verify` the agent
loop enforces could drift from the `just gate` the plan describes.

## Decision

Use **`make`** as the single task runner. The implementation plan's `just`
targets map onto `make` targets:

| Plan (`just`)     | This repo (`make`)         |
| ----------------- | -------------------------- |
| `just lint`       | `make lint`                |
| `just typecheck`  | `make typecheck` (Phase 0) |
| `just test`       | `make test`                |
| `just gate N`     | `make gate-N` (Phase 0)    |
| `just gate-all`   | `make gate-all` (Phase 0)  |

`make verify` remains the cumulative gate the agent hooks call; `make gate-all`
runs the full release-gate sequence. The `gate-N` / `gate-all` / `typecheck`
targets are added in build phase p00 (`specs/2026-07-p00-scaffold/`), not before.

This is a deviation from the implementation plan's stated tooling. The plan text
is not amended (it remains the historical build order); this ADR is the record of
record for the runner choice.

## Consequences

- One task runner, already wired into the verify gate, the hooks, and both
  supported agents — no drift between "the plan's gate" and "the gate the agent
  enforces".
- No new tool to install in dev or CI environments.
- Anyone following the implementation plan verbatim must mentally translate
  `just <x>` → `make <x>`; this ADR and the p00 plan document the mapping.
- `make`'s tab-sensitivity and weaker ergonomics vs `just` are accepted; the
  target surface is small and stable.
- If `just` is adopted later, it supersedes this ADR with a new one and the
  harness template files (`.claude/settings.json`, hooks) must switch in lockstep.
