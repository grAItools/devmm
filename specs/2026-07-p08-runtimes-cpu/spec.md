# Phase 8 — runtimes: base, discovery, CPU

> **Context.** Step 08 of the devmm v0.1 build. Authoritative spec: [`specs/devmm-design.md`](../devmm-design.md); build order: [`specs/devmm-implementation-plan.md`](../devmm-implementation-plan.md) (Phase 8). Design sections: §4. This step depends on **p07-export-tensor-empty** landing first.
> The overall v0.1 goals, non-goals and release gate live in the design doc and implementation plan; this folder is scoped to a single phase.

## Problem
The registry's default MR is a placeholder and there is no runtime discovery, so `empty()` cannot pick a sensible default per device.

## Goal
Runtime discovery + the CPU runtime exist; `empty(device="cpu")` with no explicit MR uses the runtime default, completing the CPU story.

## Users & stakeholders
Python developers building on devmm (this phase's consumers are the later phases and the test suite) and the grAItools maintainers who sign off against [`specs/devmm-design.md`](../devmm-design.md).

## Success criteria
- `runtime_names()` reports availability without importing heavyweight modules; `available_runtimes()` constructs only passing probes.
- A fake third-party runtime registered via an in-test entry point is discovered, ordered after built-ins, and loadable.
- `DEVMM_RUNTIME=cpu` forces selection; a bogus value raises `RuntimeUnavailableError` with an actionable message.
- `empty(device="cpu")` with no MR uses the runtime default and re-runs the Phase-7 round-trip suite green.

## Non-goals
- No CUDA/ROCm runtimes (Phases 9/10).

## Open questions
- None expected.
