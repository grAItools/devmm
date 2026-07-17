# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-07-17

First release: a uniform, pure-Python, zero-dependency interface for
allocating and managing device memory across CPU, CUDA and ROCm, exporting
every allocation as a DLPack ≥ 1.0 producer.

- GPU hardware test suites (CUDA T2, ROCm T3) are **waived for this release**
  per [ADR 0003](docs/adr/0003-gpu-suite-waiver-for-0.1.0.md): no GPU runner
  has executed them yet, so GPU code paths are validated only through the
  fake-API suites on CPU-only CI. The suites ship skip-gated behind the
  `gpu_cuda`/`gpu_rocm` markers (`DEVMM_GPU=cuda|rocm` opts in).

### Added

- `devmm.empty` / `devmm.empty_like` allocate a `Tensor` (a DLPack-exportable
  view over an owning, stream-ordered `DeviceBuffer`) with layout control at
  allocation time.
- `Device`/`DeviceType` (DLPack device codes verbatim), `DType` (DLPack
  `(code, bits, lanes)` with NumPy/Array-API constructors), and first-class
  `Stream`s with the `DEFAULT`/`LEGACY_DEFAULT`/`PER_THREAD_DEFAULT`
  sentinels.
- `Layout`/`LayoutPolicy` with the shipped policies `RowMajor`, `ColMajor`,
  `Permuted`, `Aligned` (stride padding + base alignment) and
  `DeviceOptimal`.
- `DeviceMemoryResource` ABC (rmm-isomorphic `allocate`/`deallocate`) with
  the `StatisticsAdaptor`, `LoggingAdaptor`, `LimitingAdaptor` stack and
  `CallbackMemoryResource`.
- Concrete memory resources: `mrs.cpu` (`BytearrayMemoryResource`,
  `MallocMemoryResource`, experimental `NumpyHandlerMemoryResource`),
  `mrs.cuda` (`CudaRuntimeMemoryResource`, `RmmMemoryResource`,
  `CupyAllocatorMemoryResource`), `mrs.rocm` (`HipRuntimeMemoryResource`,
  `HipmmMemoryResource`).
- Per-device current-MR registry: `get_current_memory_resource`,
  `set_current_memory_resource`, and the `contextvars`-scoped
  `using_memory_resource`.
- Device runtimes with lazy discovery (`available_runtimes`,
  `runtime_names`, `runtime_for`), the `DEVMM_RUNTIME` override, the
  `devmm.runtimes` entry-point group, and platform-keyed CUDA/ROCm probes
  with `rmm`-module disambiguation.
- DLPack 1.1 export: versioned + legacy capsules with version negotiation,
  read-only flag, `stream=` consumer handoff through runtime events, and
  `BufferError` refusals per the protocol.
- Integrations (explicit, restorable `install()` arrows):
  `integrations.numpy` (NEP-49 handler), `integrations.cupy`
  (`set_allocator` bridge), `integrations.numba` (`DevmmEMMPlugin`),
  `integrations.rmm` (per-device resource bridge).
- Public testing utilities: `devmm.testing.mr_conformance`,
  `devmm.testing.dlpack_conformance`, and `RecordingMemoryResource`.

[Unreleased]: https://github.com/grAItools/devmm/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/grAItools/devmm/releases/tag/v0.1.0
