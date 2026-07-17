# Phase 4 — CPU memory resources & conformance suite

> **Context.** Step 04 of the devmm v0.1 build. Authoritative spec: [`work/devmm-design.md`](../devmm-design.md); build order: [`work/devmm-implementation-plan.md`](../devmm-implementation-plan.md) (Phase 4). Design sections: §5.1. This step depends on **p03-stream-memory-resource** landing first.
> The overall v0.1 goals, non-goals and release gate live in the design doc and implementation plan; this folder is scoped to a single phase.

## Problem
No memory resource actually allocates real memory, so nothing can be written, read back, or handed to a consumer.

## Goal
`BytearrayMemoryResource` and `MallocMemoryResource` allocate real, correctly aligned CPU memory and both pass one reusable MR conformance suite on every OS.

## Users & stakeholders
Python developers building on devmm (this phase's consumers are the later phases and the test suite) and the grAItools maintainers who sign off against [`work/devmm-design.md`](../devmm-design.md).

## Success criteria
- A known pattern written into a returned pointer reads back byte-exact; adjacent allocations never alias.
- Returned pointers satisfy the requested alignment across hypothesis sizes/alignments; `guaranteed_alignment()` is honest.
- After N alloc/free pairs the MR's internal tables are empty; double-free and unknown-pointer-free raise.
- BytearrayMR pins its backing store (resize raises `BufferError`) and releases it on free (weakref dies).
- On Windows, `_aligned_malloc`'d pointers are freed only with `_aligned_free` (family tracking); zero-byte allocation round-trips.

## Non-goals
- No `empty()`-level over-allocation yet (Phase 7); `NumpyHandlerMR` is Phase 11.

## Open questions
- None expected.
