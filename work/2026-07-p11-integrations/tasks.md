# Tasks — Phase 11 — third-party MRs & integrations

Mirror of [`plan.md`](plan.md). Tick each box in the same commit as the change.
Write the tests first and observe them fail/skip before implementing. Run the
phase gate before handing off to `/verify`. Do not start the next step until this
one is verified.

## Tests first
- [x] Test (write first, observe red/skip): T0: cycle detection; `install()`/uninstall restore incl. exception path.
- [x] Test (write first, observe red/skip): (c) CPU: handler mirror vs offset table; install->stats->uninstall->prior handler; during-install arrays freeable after; range guard.
- [x] Test (write first, observe red/skip): T2: (a) CuPy pool `used_bytes()` delta + stream context; (b) Numba EMM protocol + accounted `@cuda.jit` write.

## Implement to green
- [x] Implement: Cycle detection first (T0, fakes) + reversible `install()` for every arrow.
- [x] Implement: (c) NEP-49: `PyDataMem_Handler` mirror + version offset table + range guard (CPU-only).
- [x] Implement: (a) `CupyAllocatorMR`/`integrations.cupy`; (b) Numba EMM plugin; (d) rmm bridges.

## Gate & handoff
- [x] Update the public-API snapshot: + `CupyAllocatorMemoryResource`, `NumpyHandlerMemoryResource`, `integrations.*` (per design §2).
- [x] Update `tests/traceability.md` for the contracts covered here.
- [x] Phase gate green (Gate 11 green (T0 portions everywhere; GPU portions on T2).).
- [x] Hand off to `/verify` (Reviewer) before the next step.
