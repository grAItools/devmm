# Plan — Phase 10 — ROCm runtime & MRs [GPU]

> **Context.** Step 10 of the devmm v0.1 build. Authoritative spec: [`work/devmm-design.md`](../devmm-design.md); build order: [`work/devmm-implementation-plan.md`](../devmm-implementation-plan.md) (Phase 10). Design sections: §4.2, §5.3. This step depends on **p09-cuda** landing first.
> The overall v0.1 goals, non-goals and release gate live in the design doc and implementation plan; this folder is scoped to a single phase.

## Architecture decisions
- Extract `_runtimes/_gpulib.py` so CUDA/ROCm differ only by symbol names + error tables; re-target CUDA onto it with no behaviour change. ADR: n/a (design §4.2).

## Phase — ROCm runtime & MRs [GPU]
**Scope.** Refactor the shared shim out of Phase 9, then add the HIP runtime/MRs and the platform-keyed probe + `rmm`-module disambiguation.

**Steps.**
1. Extract `_gpulib.py`; re-target CUDA onto it (no behaviour change).
2. Implement `HipRuntimeMemoryResource`, `HipmmMemoryResource`, and the disambiguation probe.

**Tests.**
- T0: fake-api suite re-parametrized for HIP; `rmm`-module disambiguation + `DEVMM_RUNTIME` override.
- T3: conformance + round-trip (CuPy/PyTorch-ROCm as available).

**Exit criteria.** Gate 10: T0 green; T3 advisory until AMD CI exists (the release gate decides its status).

**Public-API snapshot.** + `HipRuntimeMemoryResource`, `HipmmMemoryResource` (via `devmm.mrs.rocm`)

## Risks & open questions
- hipMM `rmm`-module ambiguity -> probe disambiguation + env override.
- Shim regression on CUDA -> re-run Phase-9 T0 suite after extraction.
