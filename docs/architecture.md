# Architecture

> One-page overview. Anything longer belongs in an ADR or a dedicated doc.

## High-level

`devmm` is a uniform, **pure-Python** interface for allocating, deallocating and
managing **device memory** across CPU, CUDA and ROCm. It *wraps* existing
allocators (rmm, hipMM, CuPy pools, libc) rather than implementing allocation
strategies, and exposes every allocation as a zero-copy **DLPack ≥ 1.0** producer
consumable by any Array-API library. `ctypes` is used only for building DLPack C
structs/capsules and for raw C-ABI runtime interop. The layered design (and its
rationale) is authoritative in [`work/devmm-design.md`](../work/devmm-design.md).

## Module map

- `src/devmm/` — **public re-exports only** (`empty`, `empty_like`, `Device`,
  `Stream`, `Tensor`, `DeviceBuffer`, `Layout`, `LayoutPolicy`,
  `DeviceMemoryResource`, `available_runtimes`, `runtime_for`, …).
- `src/devmm/_core/` — runtime-agnostic domain model: `device`, `stream`,
  `memory_resource` (the `DeviceMemoryResource` ABC + Statistics/Logging/
  Limiting/Callback adaptors), `buffer`, `layout`, `dtypes`, `tensor`,
  `registry` (per-device current-MR registry).
- `src/devmm/_dlpack/` — ctypes DLPack export layer: `_abi` (struct mirrors of
  `dlpack.h`) and `export` (capsule building, deleters, version negotiation,
  stream handoff).
- `src/devmm/_runtimes/` — one `DeviceRuntime` per platform (`cpu`, `cuda`, `rocm`),
  plus `base` (the SPI `Protocol`) and `_discovery` (platform probes + entry
  points).
- `src/devmm/mrs/` — **public** concrete memory resources per platform (`cpu`,
  `cuda`, `rocm`).
- `src/devmm/integrations/` — "install a devmm MR *into* X" bridges (`numpy` NEP-49,
  `cupy`, `numba` EMM plugin, `rmm`).
- `src/devmm/testing/` — mock runtime/MR and the hardware-free conformance suite.

## External dependencies

**Zero required runtime dependencies** — the CPU memory resources and the whole
DLPack layer are stdlib-only (`ctypes`). Everything else is an optional extra
(declared in `pyproject.toml`), imported lazily and probed at runtime:

- `devmm[cuda]` → `rmm` (CUDA build) — flagship stream-ordered MR.
- `devmm[rocm]` → hipMM's `rmm`-named module (ROCm).
- `devmm[cupy]` → CuPy — wrap a CuPy pool allocator as an MR, or install a devmm
  MR into CuPy.
- `devmm[numba]` → Numba — EMM plugin so Numba allocates through a devmm MR.
- `devmm[test]` → `numpy`, `array-api-strict` — exercise DLPack round-trips.

Adding a **required** runtime dependency contradicts the pure-Python, zero-dep
goal and needs an ADR.

## Boundaries

`devmm` is a library — no HTTP/RPC surface, no daemons, no scheduled jobs. Its
external boundaries are:

- **DLPack producer protocol** — `Tensor.__dlpack__` / `__dlpack_device__` are
  the outward contract consumed by array libraries; version negotiation and the
  `stream=` handoff live in `_dlpack/export.py`.
- **Native device runtimes** — `ctypes` FFI into `libcudart` / `libamdhip64`
  (and optional third-party allocator libraries), isolated behind the
  `DeviceRuntime` SPI and the `mrs/` resources.
- **Third-party allocator global state** — `integrations/*` mutate external
  state (NumPy NEP-49 handler, `cupy.cuda.set_allocator`, rmm per-device
  resource, Numba EMM). These are **always explicit calls, never import side
  effects**; each `install()` returns an `uninstall()`/context manager that
  restores the prior state.

## Generated code

None — `devmm` has no codegen pipeline. `uv.lock` is the only machine-generated
file; regenerate it with `uv lock` / `uv sync`, never hand-edit.

## See also

- ADRs of record: [`adr/`](adr/)
- Style guide: [`style.md`](style.md)
- Testing strategy: [`testing.md`](testing.md)
