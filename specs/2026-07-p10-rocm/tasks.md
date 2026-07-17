# Tasks — Phase 10 — ROCm runtime & MRs [GPU]

Mirror of [`plan.md`](plan.md). Tick each box in the same commit as the change.
Write the tests first and observe them fail/skip before implementing. Run the
phase gate before handing off to `/verify`. Do not start the next step until this
one is verified.

## Tests first
- [ ] Test (write first, observe red/skip): T0: fake-api suite re-parametrized for HIP; `rmm`-module disambiguation + `DEVMM_RUNTIME` override.
- [ ] Test (write first, observe red/skip): T3: conformance + round-trip (CuPy/PyTorch-ROCm as available).

## Implement to green
- [ ] Implement: Extract `_gpulib.py`; re-target CUDA onto it (no behaviour change).
- [ ] Implement: Implement `HipRuntimeMemoryResource`, `HipmmMemoryResource`, and the disambiguation probe.

## Gate & handoff
- [ ] Update the public-API snapshot: + `HipRuntimeMemoryResource`, `HipmmMemoryResource` (via `devmm.mrs.rocm`).
- [ ] Update `tests/traceability.md` for the contracts covered here.
- [ ] Phase gate green (Gate 10: T0 green; T3 advisory until AMD CI exists (the release gate decides its status).).
- [ ] Hand off to `/verify` (Reviewer) before the next step.
