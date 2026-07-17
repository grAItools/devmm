# 3. GPU suite waiver (T2/T3) for the 0.1.0 release gate

## Status

Proposed — requires grAItools maintainer sign-off before tagging `v0.1.0`.

## Context

The 0.1.0 release gate
([`work/devmm-implementation-plan.md`](../../work/devmm-implementation-plan.md),
Phase 12) requires:

- **Gate 2 (T2)**: the CUDA GPU job green — the MR conformance suite over
  `CudaRuntimeMR`/`RmmMR`, CuPy/PyTorch DLPack round-trips, the stream-race
  canary, and the CuPy/Numba integration suite (`tests/test_cuda_gpu.py`,
  `tests/test_integrations_gpu.py`).
- **Gate 3 (T3)**: the ROCm job green (`tests/test_rocm_gpu.py`), *"or an
  explicit signed-off waiver documenting which ROCm tests ran manually and
  where"*. The design already marks T3 advisory until AMD CI hardware is
  provisioned (Phase 10 gate).

No CUDA or ROCm CI runner is provisioned for this repository yet, and the
development host has no GPU. Both suites are fully written, hardware-gated
behind the `gpu_cuda`/`gpu_rocm` markers, and opt in via `DEVMM_GPU=cuda|rocm`
(`tests/conftest.py`); on T0/T1 they skip (75 tests currently skip on this
basis). All GPU control-flow logic is exercised on T0 through the injected
fake-api suites (`tests/test_gpu_runtime.py`, `tests/test_gpu_mrs.py`,
design §9's design-for-testability requirement), so what the waiver covers is
strictly the hardware-facing slice: real driver calls, real stream races,
real CuPy/PyTorch/rmm/Numba interop.

## Decision

Release 0.1.0 with the T2 and T3 hardware suites **waived**, on these terms:

- The suites remain required release-gate items; the waiver applies to this
  release only and does not amend the release checklist.
- No T2/T3 test may be deleted, `xfail`ed, or unhooked from its marker; the
  suites stay collectible and skip-gated so a runner can execute them
  unmodified.
- When a CUDA (T2) and/or ROCm (T3) runner becomes available, the suites run
  with `DEVMM_GPU=cuda` / `DEVMM_GPU=rocm`; any failure is treated as a
  release-blocking bug for the next tag, and this waiver is superseded.
- The manual-run record (which tests ran, hardware, driver versions) is
  appended to this file when hardware access happens; as of this writing no
  manual GPU run has been performed.

## Consequences

- 0.1.0's hardware-facing guarantees rest on the T0 fake-api suites plus the
  CPU-path DLPack oracles (NumPy, array-api-strict, the compiled ABI oracle),
  not on GPU execution. Users on CUDA/ROCm hardware are effectively the first
  T2/T3 executors; the changelog and release notes must say so.
- The `gpu_cuda`/`gpu_rocm` markers and `DEVMM_GPU` opt-in stay the single
  mechanism for hardware gating; CI gains the GPU legs without test changes
  once runners exist.
- This ADR is append-only per repository convention; running the suites on
  hardware supersedes it with a new ADR (or an appended record) rather than
  an edit to the decision.
