# 3. GPU suite waiver (T2/T3) for the 0.1.0 release gate

## Status

Accepted — signed off by the grAItools maintainer on 2026-07-17, explicitly
including the extension of the waiver to T2 (CUDA) beyond the p12 spec's
T3-only waiver clause.

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

## Manual-run record — CUDA (T2), 2026-07-18

First execution of the T2 suite on real hardware, per the "manual-run record"
term above. ROCm (T3) remains unexecuted.

- **Hardware / driver**: 4× NVIDIA GH200 120GB (aarch64), driver 590.48.01,
  system CUDA 13.1.
- **Consumers**: `rmm-cu12` 26.06.00, `cupy-cuda12x` 14.1.1,
  `torch` 2.11.0+cu128, `numba` 0.66.0 + `numba-cuda` 0.30.4, `filecheck` 1.0.3.
  Numba's JIT toolchain aligned on the cu12 wheels (`nvidia-cuda-nvcc-cu12` /
  `nvidia-nvjitlink-cu12` both 12.8.93, `CUDA_HOME` unset) — see
  [`docs/testing.md`](../testing.md#running-the-cuda-gpu-suite-on-hardware).
- **Result**: `tests/test_cuda_gpu.py` + `tests/test_integrations_gpu.py` —
  **41 passed, 1 skipped** (the stream-race misorder canary is best-effort and
  did not manifest). Reproduce with `make test-gpu-cuda`.
- **Fixes required to get green** (both against dependency versions newer than
  the suite was written for, written to work across the old and new APIs):
  the Numba EMM plugin now hands `MemoryPointer` a `ctypes.c_void_p`
  (numba-cuda ≥ 0.30 only converts that type to a driver `CUdeviceptr`); the
  rmm statistics test normalises `allocation_counts`, a `Statistics` object in
  rmm ≥ 26.06 and a dict before.

The Decision's release-blocking clause is met rather than deferred: both
failures were fixed before any tag ships, so no T2 failure is outstanding
against a release. T2 having now run on hardware, only the ROCm (T3) half of
the waiver still has work to do; a new ADR supersedes this one when T3 runs.
