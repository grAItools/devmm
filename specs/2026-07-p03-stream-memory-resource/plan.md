# Plan — Phase 3 — stream, memory resource & recording fixture

> **Context.** Step 03 of the devmm v0.1 build. Authoritative spec: [`specs/devmm-design.md`](../devmm-design.md); build order: [`specs/devmm-implementation-plan.md`](../devmm-implementation-plan.md) (Phase 3). Design sections: §3.2, §3.3. This step depends on **p02-layout** landing first.
> The overall v0.1 goals, non-goals and release gate live in the design doc and implementation plan; this folder is scoped to a single phase.

## Architecture decisions
- Adaptors hold `upstream` strongly (fixes rmm's non-owning-ref hazard at our layer). ADR: n/a (design §3.3).

## Phase — stream, memory resource & recording fixture
**Scope.** Build `RecordingMemoryResource` first (it is the suite's fixture), then the ABCs, sentinels, `CpuStream`, and the Statistics/Logging/Limiting/Callback adaptors.

**Steps.**
1. Build `testing/_recording.py` with its own tests (fake pointers, misuse detection).
2. Implement `Stream` ABC + `CpuStream` + sentinels.
3. Implement `DeviceMemoryResource` ABC (`stream_ordered`, `guaranteed_alignment()`, `available_memory()`).
4. Implement `Statistics`, `Logging`, `Limiting` adaptors and `CallbackMemoryResource`.

**Tests.**
- `RecordingMemoryResource` own tests.
- Adaptor forwarding + strong-upstream-survives-gc (weakref probe).
- `StatisticsAdaptor` under hypothesis interleavings + 8-thread x 1k soak.
- `LimitingAdaptor` boundary-exact; `CallbackMemoryResource` args + exception propagation.
- `CpuStream` no-ops; sentinel identity.

**Exit criteria.** Gate 3 green; `RecordingMemoryResource` usable as a fixture downstream.

**Public-API snapshot.** + `Stream`, `DeviceMemoryResource`, adaptors, sentinels (per design §2 public surface)

## Risks & open questions
- Statistics accounting races -> thread soak.
- Adaptor lifetime bug -> weakref chain test.
