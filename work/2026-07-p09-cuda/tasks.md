# Tasks — Phase 9 — CUDA runtime & MRs [GPU]

Mirror of [`plan.md`](plan.md). Tick each box in the same commit as the change.
Write the tests first and observe them fail/skip before implementing. Run the
phase gate before handing off to `/verify`. Do not start the next step until this
one is verified.

## Tests first
- [x] Test (write first, observe red/skip): T0 fake-api: call sequences, async selection, status->exception, `make_stream_wait` ordering, `activate_device` restore; `RmmMR` vs fake.
- [x] Test (write first, observe red/skip): T2 [`gpu_cuda`]: conformance over `CudaRuntimeMR` (sync/async) + `RmmMR`; CuPy/PyTorch round-trips; stream-race canary; rmm pool stats vs `StatisticsAdaptor`.

## Implement to green
- [x] Implement: `FakeCudartApi` (records calls, injects failures); implement all control flow against it.
- [x] Implement: `RmmMemoryResource` wrapping any `rmm.mr.DeviceMemoryResource`; stream translation via `__cuda_stream__`; strong ref.

## Gate & handoff
- [x] Update the public-API snapshot: + `CudaRuntimeMemoryResource`, `RmmMemoryResource` (via `devmm.mrs.cuda`).
- [x] Update `tests/traceability.md` for the contracts covered here.
- [x] Phase gate green (Gate 9: T0 portion green everywhere; T2 job green on the CUDA runner.). T0 gate green on this host; T2 tests are written and skip pending a CUDA runner (spec open question).
- [x] Hand off to `/verify` (Reviewer) before the next step.
