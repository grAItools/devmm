# Plan — Phase 9 — CUDA runtime & MRs [GPU]

> **Context.** Step 09 of the devmm v0.1 build. Authoritative spec: [`work/devmm-design.md`](../devmm-design.md); build order: [`work/devmm-implementation-plan.md`](../devmm-implementation-plan.md) (Phase 9). Design sections: §4, §5.2. This step depends on **p08-runtimes-cpu** landing first.
> The overall v0.1 goals, non-goals and release gate live in the design doc and implementation plan; this folder is scoped to a single phase.

## Architecture decisions
- **Design-for-testability (mandatory):** the runtime + raw MR take an injected `api` wrapping every `libcudart` call; only `api` construction is GPU-only. ADR: n/a (design/IP §9).

## Phase — CUDA runtime & MRs [GPU]
**Scope.** `_runtimes/cuda.py` + `mrs/cuda.py` (`CudaRuntimeMemoryResource`, `RmmMemoryResource`) over an injected `libcudart` `api`.

**Steps.**
1. `FakeCudartApi` (records calls, injects failures); implement all control flow against it.
2. `RmmMemoryResource` wrapping any `rmm.mr.DeviceMemoryResource`; stream translation via `__cuda_stream__`; strong ref.

**Tests.**
- T0 fake-api: call sequences, async selection, status->exception, `make_stream_wait` ordering, `activate_device` restore; `RmmMR` vs fake.
- T2 [`gpu_cuda`]: conformance over `CudaRuntimeMR` (sync/async) + `RmmMR`; CuPy/PyTorch round-trips; stream-race canary; rmm pool stats vs `StatisticsAdaptor`.

**Exit criteria.** Gate 9: T0 portion green everywhere; T2 job green on the CUDA runner.

**Public-API snapshot.** + `CudaRuntimeMemoryResource`, `RmmMemoryResource` (via `devmm.mrs.cuda`)

## Risks & open questions
- Stream-handoff ordering -> fake-api sequence assertions + GPU race canary.
- Context/async path bugs -> fake-api failure injection.
