# Phase 9 — CUDA runtime & MRs [GPU]

> **Context.** Step 09 of the devmm v0.1 build. Authoritative spec: [`specs/devmm-design.md`](../devmm-design.md); build order: [`specs/devmm-implementation-plan.md`](../devmm-implementation-plan.md) (Phase 9). Design sections: §4, §5.2. This step depends on **p08-runtimes-cpu** landing first.
> The overall v0.1 goals, non-goals and release gate live in the design doc and implementation plan; this folder is scoped to a single phase.

## Problem
devmm cannot allocate on NVIDIA GPUs, and GPU control-flow logic must be verifiable without hardware to be trustworthy.

## Goal
A CUDA runtime and `CudaRuntimeMR`/`RmmMR` built on an injected `api`, with all control flow unit-tested on T0 and full round-trips on T2.

## Users & stakeholders
Python developers building on devmm (this phase's consumers are the later phases and the test suite) and the grAItools maintainers who sign off against [`specs/devmm-design.md`](../devmm-design.md).

## Success criteria
- On T0 with a `FakeCudartApi`: alloc/free call sequences are exact; `async_alloc="auto"` selects `cudaMallocAsync` iff supported; every nonzero status maps to the right exception; `make_stream_wait` emits create->record->wait->destroy; `activate_device` restores on exception.
- `RmmMemoryResource` forwards to a fake rmm MR with correct stream translation and a strong ref to the wrapped MR.
- On T2 hardware: the CPU conformance suite passes over `CudaRuntimeMR` (sync + async) and `RmmMR`; DLPack round-trips through CuPy and PyTorch (padded strides included).
- The stream-race canary is correct with handoff enabled and demonstrably wrong with it forced off; rmm pool statistics agree with `StatisticsAdaptor` counts.

## Non-goals
- No ROCm (Phase 10); no CuPy/Numba/NumPy integrations (Phase 11).

## Open questions
- Is a CUDA (T2) CI runner available, or do T2 boxes skip until provisioned?
