# Tasks — Phase 8 — runtimes: base, discovery, CPU

Mirror of [`plan.md`](plan.md). Tick each box in the same commit as the change.
Write the tests first and observe them fail/skip before implementing. Run the
phase gate before handing off to `/verify`. Do not start the next step until this
one is verified.

## Tests first
- [x] Test (write first, observe red/skip): Probe laziness (subprocess: no heavy modules in `sys.modules`).
- [x] Test (write first, observe red/skip): Fake entry-point runtime discovered + ordered + loadable.
- [x] Test (write first, observe red/skip): `DEVMM_RUNTIME=cpu` forces; bogus -> `RuntimeUnavailableError`.
- [x] Test (write first, observe red/skip): End-to-end `empty(device="cpu")` default-MR path re-runs the Phase-7 suite.

## Implement to green
- [x] Implement: SPI + discovery + `available_runtimes()`/`runtime_names()`/`runtime_for()` + env override.
- [x] Implement: CPU runtime (memcpy via `ctypes.memmove`, default MR = `MallocMemoryResource`); update the Phase-5 sentinel test to the real default path.

## Gate & handoff
- [x] Update the public-API snapshot: + `available_runtimes`, `runtime_names`, `runtime_for` (+ CPU runtime).
- [x] Update `tests/traceability.md` for the contracts covered here.
- [x] Phase gate green (Gate 8 green — CPU feature-complete and the DLPack surface oracle-verified. Tag `v0.1.0a1`.).
- [x] Hand off to `/verify` (Reviewer) before the next step.
