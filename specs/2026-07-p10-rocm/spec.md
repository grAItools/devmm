# Phase 10 — ROCm runtime & MRs [GPU]

> **Context.** Step 10 of the devmm v0.1 build. Authoritative spec: [`specs/devmm-design.md`](../devmm-design.md); build order: [`specs/devmm-implementation-plan.md`](../devmm-implementation-plan.md) (Phase 10). Design sections: §4.2, §5.3. This step depends on **p09-cuda** landing first.
> The overall v0.1 goals, non-goals and release gate live in the design doc and implementation plan; this folder is scoped to a single phase.

## Problem
devmm cannot allocate on AMD GPUs, and the CUDA/ROCm FFI logic should not be duplicated; the `rmm`-name collision must be disambiguated.

## Goal
A shared FFI shim underpins both platforms; `HipRuntimeMR`/`HipmmMR` work and the platform probe disambiguates hipMM's `rmm` module.

## Users & stakeholders
Python developers building on devmm (this phase's consumers are the later phases and the test suite) and the grAItools maintainers who sign off against [`specs/devmm-design.md`](../devmm-design.md).

## Success criteria
- The Phase-9 fake-api suite, re-parametrized for HIP symbol names/error tables, passes on T0 (the shared shim pays off).
- Given fake `rmm` modules with CUDA-ish vs HIP-ish markers, the probe selects correctly and `DEVMM_RUNTIME` overrides win.
- On T3 hardware, the conformance + round-trip suite passes with CuPy-ROCm and/or PyTorch-ROCm as available.

## Non-goals
- Third-party integrations remain Phase 11.

## Open questions
- T3 (AMD) CI is likely unavailable initially — is an advisory/manual-waiver acceptable until hardware exists?
