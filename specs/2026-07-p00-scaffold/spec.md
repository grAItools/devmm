# Phase 0 — Scaffold delta & CI skeleton

> **Context.** Step 00 of the devmm v0.1 build. Authoritative spec: [`specs/devmm-design.md`](../devmm-design.md); build order: [`specs/devmm-implementation-plan.md`](../devmm-implementation-plan.md) (Phase 0). Design sections: §2. This is the first step; no prior step is required.
> The overall v0.1 goals, non-goals and release gate live in the design doc and implementation plan; this folder is scoped to a single phase.

## Problem
The repo has a stub package tree, `pyproject.toml`, `py.typed` and a smoke test from init, but no cumulative gate, CI, strict typing, coverage, or the always-on enforcement tests. Without them "done" is undefined and dependency creep / public-API drift go undetected.

## Goal
A green cumulative gate (lint + strict types + tests + wheel-in-bare-venv) runs locally and in CI on the T0 matrix and a T1 job, with the public-API snapshot and zero-dependency tests wired in.

## Users & stakeholders
Python developers building on devmm (this phase's consumers are the later phases and the test suite) and the grAItools maintainers who sign off against [`specs/devmm-design.md`](../devmm-design.md).

## Success criteria
- Building the wheel, installing it into an environment with no other packages, and importing it succeeds.
- The public-API snapshot test passes with an empty exported surface.
- One gate command runs lint, `mypy --strict`, tests and the packaging check and is green.
- CI runs that gate on 3 OSes x {min, max Python} plus a compiler-enabled Linux job.

## Non-goals
- No library behaviour is implemented here — the package still re-exports nothing.
- No GPU jobs in CI yet (added when the first `[GPU]` phase lands).

## Open questions
- Confirm `make` + Python 3.12 over the plan's `just` + 3.10 (see Architecture decisions).
- Confirm the T1 (compiler) CI job image and the GPU-CI timeline.
