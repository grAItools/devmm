# Tasks — Phase 8 — runtimes: base, discovery, CPU

Mirror of [`plan.md`](plan.md). Tick each box in the same commit as the change.
Write the tests first and observe them fail/skip before implementing. Run the
phase gate before handing off to `/verify`. Do not start the next step until this
one is verified.

## Tests first
- [ ] Test (write first, observe red/skip): Probe laziness (subprocess: no heavy modules in `sys.modules`).
- [ ] Test (write first, observe red/skip): Fake entry-point runtime discovered + ordered + loadable.
- [ ] Test (write first, observe red/skip): `DEVMM_RUNTIME=cpu` forces; bogus -> `RuntimeUnavailableError`.
- [ ] Test (write first, observe red/skip): End-to-end `empty(device="cpu")` default-MR path re-runs the Phase-7 suite.

## Implement to green
- [ ] Implement: SPI + discovery + `available_runtimes()`/`runtime_names()`/`runtime_for()` + env override.
- [ ] Implement: CPU runtime (memcpy via `ctypes.memmove`, default MR = `MallocMemoryResource`); update the Phase-5 sentinel test to the real default path.

## Gate & handoff
- [ ] Update the public-API snapshot: + `available_runtimes`, `runtime_names`, `runtime_for` (+ CPU runtime).
- [ ] Update `tests/traceability.md` for the contracts covered here.
- [ ] Phase gate green (Gate 8 green — CPU feature-complete and the DLPack surface oracle-verified. Tag `v0.1.0a1`.).
- [ ] Hand off to `/verify` (Reviewer) before the next step.
