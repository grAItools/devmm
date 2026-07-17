# Phase 7 — export, Tensor & empty()/empty_like() (critical)

> **Context.** Step 07 of the devmm v0.1 build. Authoritative spec: [`specs/devmm-design.md`](../devmm-design.md); build order: [`specs/devmm-implementation-plan.md`](../devmm-implementation-plan.md) (Phase 7). Design sections: §3.8, §7. This step depends on **p06-dlpack-abi** landing first.
> The overall v0.1 goals, non-goals and release gate live in the design doc and implementation plan; this folder is scoped to a single phase.

## Problem
Nothing yet produces a DLPack capsule or a user-facing tensor; this is where memory-safety and zero-copy correctness are won or lost.

## Goal
`empty()`/`empty_like()` produce a `Tensor` that NumPy (and array-api-strict) consume zero-copy, with a deleter chain that never leaks or frees memory a consumer still holds.

## Users & stakeholders
Python developers building on devmm (this phase's consumers are the later phases and the test suite) and the grAItools maintainers who sign off against [`specs/devmm-design.md`](../devmm-design.md).

## Success criteria
- For hypothesis (shape, dtype, policy) on CPU, `np.from_dlpack(t)` matches dtype/shape/values; padded layouts give byte-correct strides and mutation round-trips (genuine zero-copy).
- `max_version=(1,1)` yields a `dltensor_versioned` capsule and `None` a `dltensor` capsule; read-only + legacy raises `BufferError`; NumPy consumes both.
- Across the deleter lifecycle matrix (consumed/unconsumed/double-export/foreign-thread/shutdown), each allocation is deallocated exactly once at the right time; a 10^4-iteration refleak harness stays bounded with empty `gc.garbage`.
- Refusals raise `BufferError`: `dl_device` mismatch, `copy=True`, and (by validation table) invalid stream ints; zero-size exports `data == NULL` and NumPy accepts it.
- `empty_like` duck-types NumPy and array-api-strict arrays; `array_api_strict.from_dlpack` consumes devmm tensors.

## Non-goals
- No cross-device export, no `copy=True`, no DLPack import (all deferred); stream handoff is a CPU no-op until Phase 9.

## Open questions
- None expected; this phase gets the most review budget.
