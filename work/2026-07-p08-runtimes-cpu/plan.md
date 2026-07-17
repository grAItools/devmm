# Plan — Phase 8 — runtimes: base, discovery, CPU

> **Context.** Step 08 of the devmm v0.1 build. Authoritative spec: [`work/devmm-design.md`](../devmm-design.md); build order: [`work/devmm-implementation-plan.md`](../devmm-implementation-plan.md) (Phase 8). Design sections: §4. This step depends on **p07-export-tensor-empty** landing first.
> The overall v0.1 goals, non-goals and release gate live in the design doc and implementation plan; this folder is scoped to a single phase.

## Architecture decisions
- Probes key off platform, not module name, and are cheap; heavyweight imports are deferred to loaders. ADR: n/a (design §4.1/§4.2).

## Phase — runtimes: base, discovery, CPU
**Scope.** `DeviceRuntime` protocol + `(name, probe, loader)` registry + entry-point loading + module query API + `DEVMM_RUNTIME` override + CPU runtime; rewire the Phase-5 registry lazy default through `runtime_for(...)`.

**Steps.**
1. SPI + discovery + `available_runtimes()`/`runtime_names()`/`runtime_for()` + env override.
2. CPU runtime (memcpy via `ctypes.memmove`, default MR = `MallocMemoryResource`); update the Phase-5 sentinel test to the real default path.

**Tests.**
- Probe laziness (subprocess: no heavy modules in `sys.modules`).
- Fake entry-point runtime discovered + ordered + loadable.
- `DEVMM_RUNTIME=cpu` forces; bogus -> `RuntimeUnavailableError`.
- End-to-end `empty(device="cpu")` default-MR path re-runs the Phase-7 suite.

**Exit criteria.** Gate 8 green — CPU feature-complete and the DLPack surface oracle-verified. Tag `v0.1.0a1`.

**Public-API snapshot.** + `available_runtimes`, `runtime_names`, `runtime_for` (+ CPU runtime)

## Risks & open questions
- Eager heavyweight import -> probe-laziness subprocess test.
- Default-MR regression -> Phase-7 suite re-run through the runtime path.
