# Phase 5 — buffer & registry

> **Context.** Step 05 of the devmm v0.1 build. Authoritative spec: [`work/devmm-design.md`](../devmm-design.md); build order: [`work/devmm-implementation-plan.md`](../devmm-implementation-plan.md) (Phase 5). Design sections: §3.4, §3.5. This step depends on **p04-cpu-mrs** landing first.
> The overall v0.1 goals, non-goals and release gate live in the design doc and implementation plan; this folder is scoped to a single phase.

## Problem
Raw pointers from an MR have no ownership, lifetime safety net, or default-selection mechanism, so higher layers cannot own memory safely.

## Goal
`DeviceBuffer` owns an allocation with a GC safety net and idempotent free, and the registry resolves a per-device current MR with contextvar scoping.

## Users & stakeholders
Python developers building on devmm (this phase's consumers are the later phases and the test suite) and the grAItools maintainers who sign off against [`work/devmm-design.md`](../devmm-design.md).

## Success criteria
- `free()` deallocates exactly once on the allocation stream by default (explicit stream when given); a second `free()` is a no-op; use-after-free raises.
- Dropping all refs and collecting yields exactly one deallocation, including when the buffer is in a reference cycle; the finalizer never resurrects it.
- A process that allocates and exits without freeing terminates cleanly with no stderr traceback.
- The registry holds MRs strongly; `using_memory_resource` restores on exit and on exception; overrides are isolated across threads and asyncio tasks.
- `copy_from_host`/`copy_to_host` round-trip byte-exact on CPU MRs.

## Non-goals
- Runtime-routed memcpy and the real lazy default are Phase 8 (a sentinel default is used here).

## Open questions
- None expected.
