# Phase 3 — stream, memory resource & recording fixture

> **Context.** Step 03 of the devmm v0.1 build. Authoritative spec: [`specs/devmm-design.md`](../devmm-design.md); build order: [`specs/devmm-implementation-plan.md`](../devmm-implementation-plan.md) (Phase 3). Design sections: §3.2, §3.3. This step depends on **p02-layout** landing first.
> The overall v0.1 goals, non-goals and release gate live in the design doc and implementation plan; this folder is scoped to a single phase.

## Problem
There is no allocation interface and no test double to exercise allocator behaviour, so no MR or buffer can be built or verified.

## Goal
The `DeviceMemoryResource` ABC, `Stream`/`CpuStream`, the adaptor stack, and a `RecordingMemoryResource` fixture exist and are self-tested.

## Users & stakeholders
Python developers building on devmm (this phase's consumers are the later phases and the test suite) and the grAItools maintainers who sign off against [`specs/devmm-design.md`](../devmm-design.md).

## Success criteria
- `RecordingMemoryResource` produces deterministic fake pointers and detects double-free, foreign-free and size-mismatch.
- Adaptors forward `(nbytes, stream)` exactly and keep their upstream alive through gc.
- `StatisticsAdaptor` keeps `current == sum(live)`, `peak == max prefix`, monotone `total` under interleaved and multi-threaded ops.
- `LimitingAdaptor` is boundary-exact (on-limit ok, +1 raises `MemoryError`, failed alloc uncounted).
- `CpuStream.synchronize()`/`wait_raw()` are no-ops; sentinels have identity semantics.

## Non-goals
- No real memory yet (Phase 4); no pooling ever (design non-goal).

## Open questions
- None expected.
