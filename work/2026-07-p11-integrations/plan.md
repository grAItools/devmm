# Plan — Phase 11 — third-party MRs & integrations

> **Context.** Step 11 of the devmm v0.1 build. Authoritative spec: [`work/devmm-design.md`](../devmm-design.md); build order: [`work/devmm-implementation-plan.md`](../devmm-implementation-plan.md) (Phase 11). Design sections: §5.2-§5.4, §6. This step depends on **p10-rocm** landing first.
> The overall v0.1 goals, non-goals and release gate live in the design doc and implementation plan; this folder is scoped to a single phase.

## Architecture decisions
- Provide arrows mutate third-party global state and are always explicit calls, never import side effects; `install()` detects direct cycles and raises. ADR: n/a (design §6).

## Phase — third-party MRs & integrations
**Scope.** In order: (a) `CupyAllocatorMR` + `integrations/cupy.py`; (b) `integrations/numba.py` EMM plugin; (c) `NumpyHandlerMR` + `integrations/numpy.py` (NEP-49, CPU-only, full treatment); (d) `integrations/rmm.py` bridges.

**Steps.**
1. Cycle detection first (T0, fakes) + reversible `install()` for every arrow.
2. (c) NEP-49: `PyDataMem_Handler` mirror + version offset table + range guard (CPU-only).
3. (a) `CupyAllocatorMR`/`integrations.cupy`; (b) Numba EMM plugin; (d) rmm bridges.

**Tests.**
- T0: cycle detection; `install()`/uninstall restore incl. exception path.
- (c) CPU: handler mirror vs offset table; install->stats->uninstall->prior handler; during-install arrays freeable after; range guard.
- T2: (a) CuPy pool `used_bytes()` delta + stream context; (b) Numba EMM protocol + accounted `@cuda.jit` write.

**Exit criteria.** Gate 11 green (T0 portions everywhere; GPU portions on T2).

**Public-API snapshot.** + `CupyAllocatorMemoryResource`, `NumpyHandlerMemoryResource`, `integrations.*` (per design §2)

## Risks & open questions
- NEP-49 ABI drift across NumPy versions -> version-parametrized offset table + range guard.
- Global-state corruption by `install()` -> restore-on-uninstall + exception-path tests.
