# Phase 11 — third-party MRs & integrations

> **Context.** Step 11 of the devmm v0.1 build. Authoritative spec: [`work/devmm-design.md`](../devmm-design.md); build order: [`work/devmm-implementation-plan.md`](../devmm-implementation-plan.md) (Phase 11). Design sections: §5.2-§5.4, §6. This step depends on **p10-rocm** landing first.
> The overall v0.1 goals, non-goals and release gate live in the design doc and implementation plan; this folder is scoped to a single phase.

## Problem
devmm cannot yet consume ecosystem allocators as MRs nor install a devmm MR into NumPy/CuPy/Numba/rmm, and mixing both directions can form a cycle.

## Goal
The consume + provide arrows for CuPy, Numba, NumPy and rmm work, each `install()` is reversible, and composing both directions for one library is refused.

## Users & stakeholders
Python developers building on devmm (this phase's consumers are the later phases and the test suite) and the grAItools maintainers who sign off against [`work/devmm-design.md`](../devmm-design.md).

## Success criteria
- Composing a consume+provide pair for the same library raises; every `install()` returns an uninstaller/context manager that restores prior state, including on exception.
- The NEP-49 `PyDataMem_Handler` ctypes mirror matches a NumPy-version-parametrized offset table; install->allocate->stats-grew, uninstall restores the prior handler, arrays allocated during installation stay freeable after, and out-of-range NumPy raises cleanly.
- On T2: CuPy-pool allocation is observable via `used_bytes()` deltas with correct current-device/stream context; the Numba EMM plugin passes Numba's own EMM test protocol and a `@cuda.jit` kernel writes into devmm-accounted memory.

## Non-goals
- The Numba consumer-direction MR stays a documented recipe, not shipped code (design §5.4).

## Open questions
- Which NumPy versions to pin for the NEP-49 offset table.
