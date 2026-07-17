# Plan — Phase 5 — buffer & registry

> **Context.** Step 05 of the devmm v0.1 build. Authoritative spec: [`work/devmm-design.md`](../devmm-design.md); build order: [`work/devmm-implementation-plan.md`](../devmm-implementation-plan.md) (Phase 5). Design sections: §3.4, §3.5. This step depends on **p04-cpu-mrs** landing first.
> The overall v0.1 goals, non-goals and release gate live in the design doc and implementation plan; this folder is scoped to a single phase.

## Architecture decisions
- Finalizer captures only `(mr, ptr, nbytes, stream)`, never `self`; uses `weakref.finalize`, not `__del__`. ADR: n/a (design §3.5, §7.4).

## Phase — buffer & registry
**Scope.** Implement `DeviceBuffer` (finalizer, idempotent free, context manager, CPU-path copy helpers) and the contextvar-scoped registry with a temporary sentinel default.

**Steps.**
1. `DeviceBuffer` with the `(mr, ptr, nbytes, stream)` finalizer, `closed` guard, context manager, and `ctypes.memmove` copy helpers.
2. Registry: strong-ref `dict[Device, MR]`, `using_memory_resource` contextvar override, sentinel lazy default.

**Tests.**
- Lifecycle vs RecordingMR (deallocate-once, explicit stream, use-after-free).
- Finalizer drop/cycle/no-resurrect; interpreter-shutdown subprocess.
- Registry strong-ref; restore on exit/exception; contextvar isolation (threads + asyncio); sentinel default raises cleanly.
- `copy_from_host`/`copy_to_host` byte-exact round-trip.

**Exit criteria.** Gate 5 green.

**Public-API snapshot.** + `DeviceBuffer`, registry accessors (`get`/`set`/`using_memory_resource`)

## Risks & open questions
- Finalizer capturing `self` -> cycle test.
- Contextvar leakage across tasks -> isolation test.
