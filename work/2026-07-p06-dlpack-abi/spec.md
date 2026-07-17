# Phase 6 — DLPack ABI mirrors & compiled oracle

> **Context.** Step 06 of the devmm v0.1 build. Authoritative spec: [`work/devmm-design.md`](../devmm-design.md); build order: [`work/devmm-implementation-plan.md`](../devmm-implementation-plan.md) (Phase 6). Design sections: §7.1. This step depends on **p05-buffer-registry** landing first.
> The overall v0.1 goals, non-goals and release gate live in the design doc and implementation plan; this folder is scoped to a single phase.

## Problem
The exporter needs ctypes structs that match `dlpack.h` byte-for-byte; any silent layout mismatch corrupts memory in every consumer.

## Goal
ctypes mirrors of every DLPack struct provably match a compiled `dlpack.h` on T1 and committed per-platform snapshots on T0.

## Users & stakeholders
Python developers building on devmm (this phase's consumers are the later phases and the test suite) and the grAItools maintainers who sign off against [`work/devmm-design.md`](../devmm-design.md).

## Success criteria
- On a machine with a C compiler, every `sizeof`/`offsetof` from a compiled `dlpack.h` matches the ctypes structs.
- On compiler-less machines, the ctypes structs match committed JSON snapshots for the four target platforms.
- Setting every field of `DLManagedTensorVersioned` by name round-trips through a `memoryview` byte pattern (catches reordering).

## Non-goals
- No capsule building or export logic yet (Phase 7).

## Open questions
- Which `dlpack.h` version tag to vendor (pin to a released DLPack >= 1.0).
