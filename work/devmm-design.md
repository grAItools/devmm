# `devmm` — a cross-platform tensor memory allocator interface for Python

*Architectural design proposal, revision 2.2. Project renamed to `devmm` (Device Memory Manager; PyPI handle verified free as of 2026-07-17). Changes in r2 vs. r1: merged layout concepts (§3.6), `Backend` renamed to `DeviceRuntime` with module-level availability queries (§5), and a catalogue of concrete memory resources + a two-directional integration model (§6).*

## 1. Goals and non-goals

**Goals**

- A uniform, pure-Python interface for allocating, deallocating and managing device memory across CPU, CUDA and ROCm devices, wrapping existing allocators (rmm, hipMM, CuPy pools, libc, ...) rather than implementing allocation strategies.
- First-class, rmm-style **stream-ordered** allocation semantics.
- Expose allocations as **DLPack ≥ 1.0 producers** (`__dlpack__` / `__dlpack_device__`), zero-copy consumable via `xp.from_dlpack(...)` by any Array-API-conformant library (NumPy, CuPy, PyTorch, JAX, ...).
- **Layout control** at allocation time: dimension permutation plus stride alignment/padding (e.g., pad the innermost extent so each contiguous line starts on a 128-byte boundary).
- Pure Python core; `ctypes` only for (a) building DLPack C structs/capsules and (b) raw-runtime / C-ABI interop paths.

**Non-goals (v1)**

- The `Tensor` is *not* an array library: no arithmetic, no indexing, no partial Array API. It is a typed, shaped, DLPack-exportable view over a `DeviceBuffer`.
- No cross-device export (`__dlpack__(dl_device=...)` to a *different* device raises `BufferError`), no consumption of foreign DLPack capsules (import is a v2 feature), no host↔device transfer API beyond a minimal byte-level helper needed for testing.
- No kernels, no math, no dtype casting.

## 2. Layered architecture

```
┌────────────────────────────────────────────────────────────┐
│ Public API:  empty(), empty_like(), Device, Stream,        │
│              Tensor, DeviceBuffer, Layout, LayoutPolicy,   │
│              DeviceMemoryResource, mrs.*, integrations.*,  │
│              available_runtimes(), runtime_for(), registry │
├────────────────────────────────────────────────────────────┤
│ Core domain model (runtime-agnostic, pure Python):         │
│   device.py  stream.py  memory_resource.py  buffer.py     │
│   layout.py  tensor.py  registry.py                        │
├────────────────────────────────────────────────────────────┤
│ DLPack export layer (ctypes):                              │
│   _dlpack/_abi.py     ctypes mirrors of dlpack.h structs   │
│   _dlpack/export.py   capsule building, deleters,          │
│                       version negotiation, stream handoff  │
├────────────────────────────────────────────────────────────┤
│ Device runtimes (one per platform: streams, memcpy,        │
│ device activation, defaults):                              │
│   _runtimes/base.py   (the DeviceRuntime SPI)              │
│   _runtimes/cpu.py  _runtimes/cuda.py  _runtimes/rocm.py   │
├────────────────────────────────────────────────────────────┤
│ Memory resources (many per platform) + integrations:       │
│   mrs/cpu.py    mrs/cuda.py    mrs/rocm.py                 │
│   integrations/{numpy,cupy,numba,rmm}.py                   │
└────────────────────────────────────────────────────────────┘
```

Package layout:

```
src/devmm/
├── __init__.py             # public re-exports only
├── _core/
│   ├── device.py           # DeviceType, Device
│   ├── stream.py           # Stream protocol + sentinels
│   ├── memory_resource.py  # DeviceMemoryResource ABC + adaptors
│   ├── buffer.py           # DeviceBuffer
│   ├── layout.py           # Layout, LayoutPolicy, shipped policies
│   ├── dtypes.py           # DType <-> DLDataType mapping
│   ├── tensor.py           # Tensor (the DLPack producer)
│   └── registry.py         # per-device current-MR registry
├── _dlpack/
│   ├── _abi.py             # ctypes.Structure defs from dlpack.h
│   └── export.py           # to_capsule(), deleters, negotiation
├── _runtimes/
│   ├── base.py             # DeviceRuntime SPI (Protocol)
│   ├── _discovery.py       # probe registry, entry points
│   ├── cpu.py
│   ├── cuda.py
│   └── rocm.py
├── mrs/                    # public: users import concrete MRs from here
│   ├── cpu.py              # BytearrayMR, MallocMR, NumpyHandlerMR
│   ├── cuda.py             # RmmMR, CudaRuntimeMR, CupyAllocatorMR, PyCudaMR
│   └── rocm.py             # HipmmMR, HipRuntimeMR
├── integrations/           # "install devmm into X" direction
│   ├── numpy.py            # NEP-49 handler installation
│   ├── cupy.py             # cupy.cuda.set_allocator bridge
│   ├── numba.py            # EMM plugin
│   └── rmm.py              # registry bridging
└── testing/                # mock runtime/MR, protocol conformance suite
```

## 3. Core concepts

### 3.1 `DeviceType` and `Device`

`DeviceType` is an `IntEnum` whose values are **exactly** the DLPack `DLDeviceType` codes, so no translation table is ever needed:

```python
class DeviceType(enum.IntEnum):
    CPU          = 1   # kDLCPU
    CUDA         = 2   # kDLCUDA
    CUDA_HOST    = 3   # kDLCUDAHost   (pinned; possible v1.x extension)
    ROCM         = 10  # kDLROCM
    ROCM_HOST    = 11  # kDLROCMHost
    CUDA_MANAGED = 13  # kDLCUDAManaged

@dataclasses.dataclass(frozen=True, slots=True)
class Device:
    type: DeviceType
    index: int = 0

    @classmethod
    def from_string(cls, s: str) -> "Device": ...   # "cuda:1", "rocm:0", "cpu"

    def __dlpack_device__(self) -> tuple[int, int]:
        return (int(self.type), self.index)
```

Design choice: **no ambient "current device" as a correctness mechanism.** Every buffer, stream and memory resource carries an explicit `Device`. A convenience context manager (`with device: ...`) is provided that also flips the runtime's native active device (needed because rmm/HIP semantics require the correct device active during alloc/dealloc/kernel work), but all core APIs take the device explicitly. This avoids rmm's documented Python↔C++ "current resource" divergence problems and plays better with multi-GPU HPC codes.

### 3.2 `Stream`

Streams are first-class. The abstraction is deliberately thin — an opaque handle plus ordering primitives — because the library never launches kernels:

```python
class Stream(abc.ABC):
    device: Device
    @property
    @abc.abstractmethod
    def handle(self) -> int: ...            # native cudaStream_t / hipStream_t as int
    @abc.abstractmethod
    def synchronize(self) -> None: ...
    @abc.abstractmethod
    def wait_raw(self, other_handle: int) -> None:
        """Make `self` wait on work currently enqueued on a raw foreign
        stream handle (event record + stream-wait-event). Used for the
        __dlpack__(stream=...) consumer handoff."""
    def __cuda_stream__(self):              # cuda.core stream protocol
        return (0, self.handle)
```

Per-runtime factories: `runtime.default_stream(device)`, `runtime.create_stream(device)`, `runtime.wrap_stream(device, obj_or_int)`. `wrap_stream` accepts raw ints and objects exposing `__cuda_stream__` (so `cuda.core`, CuPy and rmm streams interoperate directly, matching rmm's adoption of the CUDA stream protocol). The CPU runtime has a single no-op `Stream` whose `synchronize`/`wait_raw` do nothing — this keeps every code path uniform without `if device.type == CPU` branches in the core.

Sentinels `DEFAULT`, `LEGACY_DEFAULT`, `PER_THREAD_DEFAULT` mirror rmm's and map to the platform-specific magic handles inside each runtime.

### 3.3 `DeviceMemoryResource`

The central abstraction, deliberately isomorphic to rmm's so the mental model (and rmm docs) transfer:

```python
class DeviceMemoryResource(abc.ABC):
    device: Device

    @abc.abstractmethod
    def allocate(self, nbytes: int, stream: Stream) -> int:
        """Return device pointer (int). Raise MemoryError on failure.
        Stream-ordered contract: the memory is usable on `stream`
        immediately, elsewhere only after synchronization."""

    @abc.abstractmethod
    def deallocate(self, ptr: int, nbytes: int, stream: Stream) -> None: ...

    # Capability probes
    @property
    def stream_ordered(self) -> bool:
        """True if (de)allocation is genuinely stream-ordered. Synchronous
        MRs (malloc, cudaMalloc, ...) return False; they must still be
        SAFE under the stream-ordered calling convention: allocate() may
        ignore `stream`; deallocate() must not release memory that could
        still be in use on any stream (e.g., cudaFree/hipFree perform an
        implicit sync; libc free needs no ordering)."""
        return False

    def guaranteed_alignment(self) -> int: ...              # bytes; 1 if unknown
    def available_memory(self) -> tuple[int, int] | None:   # (free, total)
        return None
```

Signature is identical (modulo our `Stream` type) to rmm's `allocate(nbytes, stream)` / `deallocate(ptr, nbytes, stream)`, which is also what hipMM's Python layer exposes since it is a direct RMM port. The `stream_ordered` and `guaranteed_alignment` probes exist because the concrete MR catalogue (§6) now spans allocators with very different semantics; `empty()` uses `guaranteed_alignment()` to decide whether base-alignment over-allocation is needed.

**Adaptors implemented at our layer, not delegated to rmm's**, so they work uniformly across all MRs: `StatisticsAdaptor`, `LoggingAdaptor`, `LimitingAdaptor(limit_bytes)`, plus `CallbackMemoryResource(alloc_fn, dealloc_fn, device)` as the pure-Python escape hatch. Pooling is explicitly *out of scope* (configure rmm/hipMM/CuPy pools underneath instead). Each adaptor stores its `upstream` as a strong reference — this deliberately fixes at our layer the lifetime hazard rmm documents (C++ `device_buffer` holds only a non-owning resource ref and the user must keep the MR alive).

### 3.4 Per-device current-resource registry

```python
get_current_memory_resource(device) -> DeviceMemoryResource
set_current_memory_resource(mr) -> None            # keyed by mr.device
using_memory_resource(mr) -> ContextManager        # scoped override (contextvars)
```

Implementation notes: a plain `dict[Device, DeviceMemoryResource]` holding **strong references** (again, the rmm lesson: rmm's Python layer maintains its own dict precisely because the C++ `set_current_device_resource` map stores raw pointers with no lifetime provenance). The scoped override uses `contextvars.ContextVar` so it composes with async code and thread pools. Lazy default: first access for a device asks the runtime for its default MR (§5.1). Setting the CUDA current MR here does *not* call `rmm.mr.set_current_device_resource` (we don't want to mutate rmm's global state behind the user's back); the explicit bridges in `integrations.rmm` link the two registries for users who want that.

### 3.5 `DeviceBuffer`

An owning, untyped, stream-ordered allocation:

```python
class DeviceBuffer:
    ptr: int          # device pointer (0 allowed for size 0)
    nbytes: int
    device: Device
    stream: Stream    # allocation stream; also default deallocation stream
    mr: DeviceMemoryResource   # strong reference

    def free(self, stream: Stream | None = None) -> None: ...
    def __enter__/__exit__: ...   # deterministic scope-based release
```

Lifetime rules:

1. Deallocation is **stream-ordered on the allocation stream** by default; `free(stream=...)` lets callers deallocate on a different stream when they know the dependency structure (same contract as rmm).
2. GC safety-net via `weakref.finalize` (not `__del__`): robust against reference cycles and interpreter shutdown ordering. The finalizer captures only `(mr, ptr, nbytes, stream_handle)` — never `self`.
3. Idempotent `free()`; use-after-free raises through a `closed` flag checked at export time.
4. `DeviceBuffer` holds `mr` strongly, and `mr` (if an adaptor) holds its upstream strongly, so the full allocator chain outlives every allocation — including allocations kept alive only by exported DLPack capsules (§7.3).
5. Minimal `copy_from_host(data: bytes | memoryview, stream)` / `copy_to_host(stream) -> bytes` byte-level helpers backed by the runtime's `memcpy` primitive. These exist because the library is untestable without them; they are documented as testing/bootstrap utilities, not a transfer API.

### 3.6 `Layout` and `LayoutPolicy`

Two concepts (down from three in r1). A `LayoutPolicy` is a callable object that, given `(shape, dtype, device)`, produces a **concrete, resolved** `Layout`; the `Layout` keeps a reference to the policy that created it:

```python
class LayoutPolicy(abc.ABC):
    """Immutable, hashable. Instances are configuration objects."""

    @property
    @abc.abstractmethod
    def base_alignment(self) -> int:
        """Bytes. For composite/dispatching policies: the MAXIMUM base
        alignment the policy may request for any (shape, dtype, device)."""

    @property
    @abc.abstractmethod
    def unit_stride_alignment(self) -> int:
        """Bytes; same upper-bound semantics. 1 means 'no padding'."""

    @abc.abstractmethod
    def __call__(self, shape: tuple[int, ...], dtype: DType,
                 device: Device) -> "Layout": ...


@dataclasses.dataclass(frozen=True, slots=True)
class Layout:
    """A concrete, fully resolved memory layout for one (shape, dtype)."""
    permutation: tuple[int, ...]   # dim order, outermost -> innermost;
                                   # C order for ndim=3 is (0, 1, 2), F order (2, 1, 0)
    strides: tuple[int, ...]       # IN ELEMENTS (DLPack convention, not bytes!)
    required_nbytes: int           # includes padding
    base_alignment: int            # bytes actually required of the base pointer
    policy: LayoutPolicy | None = None   # provenance; None for hand-built layouts

    @property
    def is_contiguous(self) -> bool: ...
    def validate(self, shape: tuple[int, ...], itemsize: int) -> None: ...
```

Notes on this shape of the design:

- `Layout` is frozen and carries provenance. The `policy` back-reference makes `empty_like`-style reproduction and debugging trivial ("which policy produced this padding?") and costs nothing since policies are immutable singletons/configs. It is `None` for layouts constructed directly from explicit strides (e.g., the future DLPack-import path), so provenance is best-effort by design.
- The alignment **properties on the policy are declared upper bounds**, not exact per-call values (the exact values live on the produced `Layout`). This is the price of merging: a dispatching policy like `DeviceOptimal` chooses alignment per device, so its properties answer "at most 256 / at most 128". For all non-composite shipped policies the property equals the per-call value. Tests assert `layout.base_alignment <= policy.base_alignment` as an invariant.
- Since `Layout` is frozen and hashes, `policy` must be hashable too — hence the "immutable configuration object" requirement on `LayoutPolicy` implementations (frozen dataclasses in practice).

Resolution algorithm (inside each policy's `__call__`, shared via a helper): order dims by `permutation`, innermost gets stride 1; each next stride is `prev_stride * padded_extent(prev_dim)`, where the innermost extent is padded up so `padded_extent * itemsize % unit_stride_alignment == 0`. `required_nbytes = (1 + Σ strides[i] * (shape[i] - 1)) * itemsize`, rounded up to `base_alignment`. Allocation-time over-alignment is applied only when `mr.guaranteed_alignment() < layout.base_alignment` (rmm/hipMM/CUDA guarantee ≥256 B, `posix_memalign` is exact, so this is usually a no-op).

Shipped policies (all immutable callables; compose freely):

```python
RowMajor()                       # perm = identity, no padding
ColMajor()                       # perm = reversed identity
Permuted(perm)
Aligned(inner: LayoutPolicy, unit_stride_alignment=128, base_alignment=256)
DeviceOptimal()                  # GPU -> Aligned(RowMajor(), 128, 256)
                                 # CPU -> Aligned(RowMajor(), cacheline, cacheline)
```

The padded case yields non-contiguous DLPack strides. That is fully legal DLPack; note in the docs that some consumers (e.g., JAX today) may copy non-contiguous imports while others (PyTorch, CuPy) take strided views — this is precisely the observable difference between `Aligned(...)` and plain `RowMajor()`.

Deliberate limits: strides produced by policies are always positive, non-overlapping, and derivable from a permutation + padding (no broadcast/negative/overlapping strides). `Layout.validate()` enforces the same on hand-built layouts. This keeps `required_nbytes` well-defined and the exporter honest.

### 3.7 `DType`

A tiny frozen dataclass mapping 1:1 onto `DLDataType` `(code, bits, lanes)` with the standard aliases (`float32`, `float64`, `int32`, `bool_`, `bfloat16`, `complex64`, ...). Constructor accepts NumPy dtypes *duck-typed* (via `.kind`/`.itemsize`) and Array API dtype strings, without importing NumPy.

### 3.8 `Tensor`

A minimal DLPack producer — dtype + shape + strides + offset over a `DeviceBuffer`, with exactly the two protocol methods plus introspection properties:

```python
class Tensor:
    buffer: DeviceBuffer     # strong ref
    dtype: DType
    shape: tuple[int, ...]
    layout: Layout           # strides (elements) + provenance
    offset: int = 0          # elements
    read_only: bool = False  # -> DLPACK_FLAG_BITMASK_READ_ONLY

    @property
    def device(self) -> Device: ...
    @property
    def strides(self) -> tuple[int, ...]: ...   # = layout.strides

    def __dlpack_device__(self) -> tuple[int, int]: ...
    def __dlpack__(self, *, stream=None, max_version=None,
                   dl_device=None, copy=None) -> PyCapsule: ...
```

No `__getitem__`, no ops, no `.T` (a `permute_view()` free function can trivially produce a new `Tensor` sharing the buffer if it later proves useful — it's metadata-only).

Top-level factories:

```python
def empty(shape, dtype, *, device=Device(DeviceType.CPU),
          layout: Layout | LayoutPolicy = DeviceOptimal(),
          mr: DeviceMemoryResource | None = None,     # default: registry lookup
          stream: Stream | None = None) -> Tensor: ...

def empty_like(obj, *, ...) -> Tensor   # reads obj.__dlpack_device__() + Array API
                                        # .shape/.dtype duck-typing; no import of the
                                        # producing library
```

`empty` is the whole user story in one call: policy → `Layout` (or `layout.validate(shape, itemsize)` if a concrete `Layout` was passed) → `mr.allocate(layout.required_nbytes, stream)` → `DeviceBuffer` → `Tensor`. Consumption is then `xp = array_api_compat.get_namespace(...); a = xp.from_dlpack(t)`.

## 4. Device runtimes

### 4.1 The `DeviceRuntime` SPI (`_runtimes/base.py`)

Renamed from r1's `Backend`. The rename is not cosmetic: with the MR catalogue of §6, allocator *libraries* (rmm, CuPy, PyCUDA, libc) are memory-resource providers, and there can be many per platform. The SPI object is the thing there is exactly **one** of per platform — the *device runtime*: device enumeration, streams/events, memcpy, native device activation, and a default-MR policy. Names like `AllocatorLibrary` or `MemoryManager` would misdescribe it (it neither allocates nor manages memory) and collide conceptually with `DeviceMemoryResource`.

```python
class DeviceRuntime(Protocol):
    name: str                                  # "cpu", "cuda", "rocm"
    device_types: frozenset[DeviceType]
    def device_count(self, device_type: DeviceType) -> int: ...
    def default_memory_resource(self, device: Device) -> DeviceMemoryResource: ...
    def default_stream(self, device: Device) -> Stream: ...
    def create_stream(self, device: Device) -> Stream: ...
    def wrap_stream(self, device: Device, obj: object) -> Stream: ...
    def make_stream_wait(self, consumer_handle: int, producer: Stream) -> None: ...
    def memcpy(self, dst: int, src: int, nbytes: int,
               kind: CopyKind, stream: Stream) -> None: ...
    def activate_device(self, device: Device) -> ContextManager: ...
```

There is deliberately **no `is_available()`** on the class. Availability is a property of the environment, answered *before* construction; an instantiated runtime is available by construction. Discovery holds `(name, probe, loader)` triples where `probe: Callable[[], bool]` is cheap (driver-library loadability, see §4.2) and `loader` performs the heavyweight import only on demand.

Module-level query API:

```python
devmm.available_runtimes() -> tuple[DeviceRuntime, ...]   # probes all, loads passing ones
devmm.runtime_names() -> tuple[str, ...]                  # probes only, no imports
devmm.runtime_for(device: Device | DeviceType | str) -> DeviceRuntime  # loads one
```

`runtime_names()` exists so that "what's available?" never pays import costs; `available_runtimes()` is the convenience that does. Third-party runtimes register via the `devmm.runtimes` entry-point group (Metal/oneAPI later without touching core).

Default-MR chains per runtime (used by the registry's lazy default, §3.4): CPU → `MallocMR`; CUDA → `RmmMR` if rmm imports, else `CudaRuntimeMR` (async variant when the driver supports `cudaMallocAsync`); ROCm → `HipmmMR` if available, else `HipRuntimeMR`.

### 4.2 Discovery and the `rmm`-name collision

The awkward reality that **hipMM's Python package also installs a module named `rmm`** (it's a straight port that keeps RAPIDS naming) means `import rmm` is ambiguous. Probes therefore key off the *platform*, not the module name: the CUDA probe checks that an NVIDIA driver is loadable (`libcuda` via `ctypes.util.find_library` / `nvidia-ml`), the ROCm probe checks `libamdhip64`; each runtime then verifies whether the `rmm` module it imports actually targets its platform (cheap sentinel: attempt a 0-byte allocate on device 0, or inspect build metadata where exposed). Since a single environment realistically contains only one of the two, this is robust in practice, and `DEVMM_RUNTIME=cuda|rocm|cpu` provides an explicit override for the pathological cases. The ROCm runtime degrades gracefully across MRs: hipMM Python → `hip-python` → raw `ctypes` on `libamdhip64` (hipMM's Python port is still being completed upstream, so the fallback path is not hypothetical).

## 5. Concrete memory resources (`devmm.mrs`)

The catalogue below is the concrete deliverable set for v1, with per-MR semantics. Common conventions: every MR carries `device`, declares `stream_ordered` and `guaranteed_alignment()`, and is safe under the stream-ordered calling convention per the §3.3 contract.

### 5.1 CPU (`mrs/cpu.py`)

**`BytearrayMemoryResource`** — 100% pure Python, zero `ctypes` FFI. `allocate` creates `ba = bytearray(nbytes + pad)` and pins it via `c = (ctypes.c_char * len(ba)).from_buffer(ba)` (the buffer-export blocks resizing/relocation for the export's lifetime), returning `ctypes.addressof(c)` rounded up to the requested alignment; an internal `dict[ptr, (ba, c)]` keeps both alive until `deallocate` drops them. `stream_ordered=False`, `guaranteed_alignment()=1` (alignment achieved by over-allocate + offset). Value: the paranoid default that works on any CPython anywhere, and the reference MR for the conformance suite.

**`MallocMemoryResource`** — libc via stdlib `ctypes`: `posix_memalign`/`free` on POSIX, `_aligned_malloc`/`_aligned_free` on Windows (they are *not* free-compatible — the MR must remember which family allocated). Exact alignment support, so `guaranteed_alignment()` reports the MR's configured alignment and `empty()` never over-allocates. Default CPU MR.

**`NumpyHandlerMemoryResource`** *(experimental)* — allocates through **NumPy's currently installed NEP-49 data-memory handler**: mirror the `PyDataMem_Handler` struct in `ctypes`, obtain the current handler by calling `PyDataMem_GetHandler` reached through the slots of NumPy's exported `_ARRAY_API` function table (the symbols themselves have hidden visibility, so DLL symbol lookup is not an option), then invoke `handler.allocator.malloc(ctx, size)` / `.free(ctx, ptr, size)`. Value: allocations inherit whatever the process configured for NumPy (tracemalloc domain tracking, user handlers), and byte-identical allocation behavior with NumPy arrays. Marked experimental because it rides on a C API reached through the `_ARRAY_API` table (whose host module path differs between NumPy 1.x `numpy.core._multiarray_umath` and 2.x `numpy._core._multiarray_umath`) — the test suite pins supported NumPy ranges. Note the *mirror* direction — making NumPy allocate through a devmm MR — is a separate, arguably more useful integration (§6).

### 5.2 CUDA (`mrs/cuda.py`)

**`RmmMemoryResource(inner, device)`** — the flagship wrapper: holds a strong ref to any `rmm.mr.DeviceMemoryResource` and forwards `allocate/deallocate`, translating our `Stream` → `rmm.pylibrmm.stream.Stream` via the CUDA stream protocol. `stream_ordered=True`, `guaranteed_alignment()=256`. `mrs.cuda.rmm_current(device)` wraps `rmm.mr.get_per_device_resource(device.index)` so users who already configured rmm (pools, managed memory, `reinitialize`) get that configuration for free.

**`CudaRuntimeMemoryResource(device, *, async_alloc="auto")`** — plain `cudaMalloc`/`cudaFree` (and `cudaMallocAsync`/`cudaFreeAsync` when the driver supports them) via `ctypes` on `libcudart`, **no third-party dependency**. This is the "just cudaMalloc" MR. *Design deviation from the proposal:* the request was to implement this via **PyCUDA**; I recommend `ctypes`-on-`libcudart` instead, for three reasons: (1) symmetry — the ROCm fallback already does exactly this on `libamdhip64`, so the two runtimes share one small FFI shim pattern; (2) PyCUDA is a driver-API library with its own context-stack discipline, and mixing its contexts with the runtime-API primary context used by rmm/CuPy/torch is a classic source of `invalid context` bugs; (3) a compiled dependency purely to reach two C functions contradicts the pure-Python goal. PyCUDA users are still served — see next item. `stream_ordered` is `True` only on the async path; the sync path relies on `cudaFree`'s implicit synchronization for safety.

**`PyCudaMemoryResource(allocator=None, device=...)`** *(optional interop wrapper)* — for codebases already living in PyCUDA: wraps `pycuda.driver.mem_alloc` or a `pycuda.tools.DeviceMemoryPool().allocate`, keeping the returned `DeviceAllocation` objects in a `ptr → allocation` dict (freeing = dropping the ref). Documented constraints: the correct PyCUDA context must be current around calls (recommend `pycuda.autoprimaryctx`, which binds PyCUDA to the runtime-API primary context and makes it coexist with rmm/CuPy); allocation is not stream-ordered (`stream_ordered=False`). Included because the adapter is ~30 lines and real PyCUDA codebases exist; *not* the default plain-CUDA path for the reasons above.

**`CupyAllocatorMemoryResource(allocator=None, device=...)`** — wraps any **CuPy-compatible allocator**, i.e. a callable `f(nbytes) -> cupy.cuda.MemoryPointer` (`cupy.cuda.alloc`, a `MemoryPool().malloc`, or user allocators). Semantics that need care: CuPy pools key cached blocks by the *thread-local current stream*, so `allocate` runs inside `with cupy.cuda.Device(device.index), our_stream_as_cupy_ExternalStream:`; the returned `MemoryPointer` is stashed in a `ptr → MemoryPointer` dict and `deallocate` drops it (CuPy frees/returns-to-pool on refcount zero, with its own stream-safety rules — so `stream_ordered=True` for pool allocators). This MR is the bridge for teams whose GPU memory budget is already governed by a CuPy pool.

### 5.3 ROCm (`mrs/rocm.py`)

**`HipmmMemoryResource`** — same shape as `RmmMemoryResource`, wrapping hipMM's Python `rmm`-named module (§4.2 disambiguation applies). **`HipRuntimeMemoryResource`** — `hipMalloc`/`hipFree` (+ `hipMallocAsync` when available) via `ctypes` on `libamdhip64`; the fallback while hipMM's Python port completes, and the mirror image of `CudaRuntimeMemoryResource`.

### 5.4 Numba EMM: which direction?

The proposal lists "wrapper around Numba EMM plugin" as an MR. Evaluated honestly: an MR *consuming* an EMM plugin (calling `plugin.memalloc(nbytes)` and dropping the returned finalizable `MemoryPointer`) is implementable but low-value — EMM plugins are written to be plugged *into* Numba, and standalone EMM plugin instances one would want to allocate from outside Numba are rare. The high-value direction is the reverse, with rmm as precedent (`rmm.allocators.numba`): a `devmm.integrations.numba.DevmmEMMPlugin(numba.cuda.BaseCUDAMemoryManager)` that makes **Numba allocate through the current devmm MR**, giving Numba kernels pooled/tracked memory and making `devmm` statistics cover Numba allocations. v1 ships the plugin (provider direction); the consumer-direction MR is documented as a recipe on top of `CallbackMemoryResource` for the rare user who needs it.

## 6. Integrations: a two-directional model

The §5 catalogue makes a pattern explicit that is worth naming in the architecture: for every ecosystem allocator there are two possible arrows, and they are different features:

| library | consume: use *its* allocator as a devmm MR | provide: install a devmm MR *into it* |
|---|---|---|
| rmm / hipMM | `RmmMR` / `HipmmMR` (§5.2/§5.3) | `integrations.rmm.install(mr)` → `rmm.mr.set_per_device_resource` via rmm's `CallbackMemoryResource` |
| CuPy | `CupyAllocatorMR` | `integrations.cupy.install(mr)` → `cupy.cuda.set_allocator` (mirror of `rmm_cupy_allocator`) |
| NumPy | `NumpyHandlerMR` (experimental) | `integrations.numpy.install(mr)` → build a `PyDataMem_Handler` from `ctypes.CFUNCTYPE` thunks over the MR and `PyDataMem_SetHandler` it (NEP-49) |
| Numba | (recipe only, §5.4) | `integrations.numba.DevmmEMMPlugin` |
| PyCUDA | `PyCudaMR` | — (PyCUDA has no global allocator hook worth targeting) |

"Consume" arrows live in `devmm.mrs.*` and produce `DeviceMemoryResource`s; "provide" arrows live in `devmm.integrations.*`, mutate third-party global state, and therefore are **always explicit calls, never side effects of import or of `set_current_memory_resource`**. Each `install()` returns an object whose `uninstall()`/context-manager restores the previous state. The provide arrows are where the library's adaptor stack pays off: `integrations.numpy.install(StatisticsAdaptor(MallocMR(...)))` yields allocation statistics over every NumPy array in the process.

Warning documented prominently: composing both arrows for the same library (e.g. `CupyAllocatorMR` as current MR *while* `integrations.cupy.install` points CuPy at devmm) creates a cycle; `install()` functions detect the direct case and raise.

## 7. DLPack export layer (`_dlpack/`)

### 7.1 ctypes ABI mirrors (`_abi.py`)

Faithful `ctypes.Structure` definitions of `DLDevice`, `DLDataType`, `DLTensor`, `DLManagedTensor` (legacy) and `DLManagedTensorVersioned` (with `DLPackVersion version`, `void* manager_ctx`, `deleter`, `uint64_t flags`, then `dl_tensor`), kept field-for-field with `dlpack.h`, plus the two flag constants (`READ_ONLY = 1`, `IS_COPIED = 2`). One version-pinned header, unit-tested against `ctypes.sizeof`/`offsetof` expectations for both 64-bit ABIs we care about.

### 7.2 Capsule construction and the deleter (`export.py`)

Per capsule we make **one** `ctypes` allocation containing `[DLManagedTensorVersioned | shape int64[ndim] | strides int64[ndim]]` so the deleter frees a single block. Ownership chain:

1. A small `_Holder` Python object strongly references the `Tensor` (hence buffer, hence MR chain).
2. `manager_ctx` stores a pointer to the holder whose refcount we bumped via `ctypes.pythonapi.Py_IncRef`.
3. The deleter is a module-level `ctypes.CFUNCTYPE(None, POINTER(DLManagedTensorVersioned))` kept alive in a module global (critical: a garbage-collected CFUNCTYPE thunk is a segfault). ctypes callbacks **acquire the GIL automatically** when invoked from foreign threads — exactly the property the DLPack spec's deleter requirements demand, and a genuine advantage of the pure-Python/ctypes approach over hand-rolled C.
4. The deleter guards against post-finalization invocation (`Py_IsInitialized` check via `ctypes.pythonapi`), decrefs the holder, frees the block.
5. The capsule itself is created via `ctypes.pythonapi.PyCapsule_New` with a capsule-destructor that runs the managed deleter **iff** the capsule was never consumed (name still `"dltensor_versioned"`, not renamed to `"used_dltensor_versioned"`), matching the spec's reference implementation.

### 7.3 Protocol semantics implemented

- **Version negotiation**: producer max is DLPack `(1, 1)`. `max_version >= (1, 0)` (or same major) → `DLManagedTensorVersioned` in a `"dltensor_versioned"` capsule; `max_version is None` or `< (1, 0)` → legacy `DLManagedTensor` in a `"dltensor"` capsule (supporting both during the transition is what the spec recommends and what NumPy/PyTorch do). `read_only=True` + a legacy-only consumer → `BufferError` (legacy struct can't express the flag; exporting silently mutable is unsafe).
- **`stream=` handoff** (the reason streams are first-class): consumer passes a raw stream handle. Producer behavior: `stream == -1` → no synchronization; `stream is None` on CPU → nothing; otherwise validate per-platform conventions (CUDA: `0` disallowed, `1` = legacy default, `2` = per-thread default; ROCm: `0` = default) and call `runtime.make_stream_wait(consumer_handle, producer_stream)` — event record + stream-wait-event, falling back to `producer_stream.synchronize()` if events are unavailable. The producer stream is `tensor.buffer.stream`.
- **`dl_device=`**: `None` or equal to `__dlpack_device__()` → proceed; anything else → `BufferError` (no cross-device export in v1; the message points users at copying via their array library).
- **`copy=`**: `False` or `None` → zero-copy export; `True` → `BufferError` in v1 (a same-device copy path needs a memcpy we already have, so this is an easy v1.x follow-up: allocate from the same MR, `memcpy` D2D, set `IS_COPIED`).
- Zero-size tensors export `data = NULL`; contiguous tensors may still pass explicit strides (always-explicit is simpler and legal; `strides=NULL` shortcut is an optimization we skip).

### 7.4 Lifetime diagram

```
consumer array (torch/cupy/...) ── capsule deleter ─▶ _Holder ─▶ Tensor
Tensor ─▶ DeviceBuffer ─▶ DeviceMemoryResource ─▶ (adaptor upstream ...) ─▶ wrapped rmm/cupy/libc allocator
DeviceBuffer.finalizer ─▶ (mr, ptr, nbytes, stream)   # safety net only
registry ─▶ strong refs to current MRs
```

Invariant: memory can never outlive its allocator chain, and exported memory can never be freed while any consumer holds it — even if the user drops every `devmm` reference.

## 8. Errors, threading, misc

- `MemoryError` from `allocate` failures (with device + MR chain in the message), `BufferError` for all `__dlpack__` refusals (per spec), `RuntimeUnavailableError(ImportError)` for missing runtimes, `StreamError` for handoff failures.
- Thread-safety: the registry and discovery are lock-protected; MRs themselves inherit the thread-safety of what they wrap (rmm MRs are thread-safe per-resource; libc malloc is; `BytearrayMR`/`CupyAllocatorMR` protect their internal dicts with a lock). Adaptors that mutate state (`Statistics`, `Limiting`) use a lock.
- Logging via stdlib `logging` (`devmm.mr`, `devmm.dlpack` loggers); `LoggingAdaptor` emits structured records.
- Typing: fully typed, `py.typed`; `Protocol`-based SPI keeps runtimes duck-typed and independently distributable.
- Packaging: zero required dependencies (`BytearrayMR`, `MallocMR` and the whole DLPack layer are stdlib-only). Extras: `devmm[cuda]` → `rmm-cuXX`, `devmm[rocm]` → `amd-hipmm` (best effort given its index situation), `devmm[cupy]`, `devmm[numba]`, `devmm[test]` → numpy + array-api-strict.

## 9. Testing strategy

- The **CPU MRs make the whole protocol testable without GPUs**: `numpy.from_dlpack(devmm.empty(...))` round-trips exercise capsule construction, versioned/legacy negotiation (NumPy supports both), deleter invocation (checked via `weakref` + gc), read-only flags, and padded-stride imports — run against *both* `BytearrayMR` and `MallocMR` to catch pinning/alignment bugs. `array-api-strict` validates consumer-side conformance.
- Property-based tests (hypothesis) for every shipped `LayoutPolicy`: strides are a valid permutation-derived set, `required_nbytes` bounds every addressable element, alignment postconditions hold, and `layout.base_alignment <= policy.base_alignment` (the upper-bound invariant of §3.6).
- A `testing.MockRuntime` + recording MR asserts stream-ordering contracts (alloc/dealloc stream pairing, handoff event sequencing) without hardware.
- Integration round-trips per optional dependency, gated by availability: NEP-49 install/uninstall restores the prior handler; `CupyAllocatorMR` returns memory to the originating pool; the Numba EMM plugin passes Numba's EMM test hooks.
- GPU CI (when available): one smoke job per platform doing rmm-pool + torch/cupy `from_dlpack` round trips and a stream-race canary (write on stream A, consume via `__dlpack__(stream=B)`, verify no torn reads with events disabled vs enabled).
- ABI struct tests: `ctypes.sizeof(DLManagedTensorVersioned)`, `PyDataMem_Handler`, etc. against constants captured from compiled references, guarding against silent field drift.

## 10. Deferred (v2 candidates)

Consuming external DLPack tensors (`devmm.from_dlpack`, becoming a consumer — note this is where `Layout(policy=None)` provenance slots in); same-device `copy=True` export and cross-device `dl_device=` export via `memcpy`; pinned-host (`CUDA_HOST`/`ROCM_HOST`) and managed-memory resources as first-class device types; the new C-level `DLPackExchangeAPI` fast path (`__dlpack_c_exchange_api__`) for capsule-free exchange — notable because its allocator hook (`DLPackManagedTensorAllocator`) would let *other* libraries allocate through `devmm`'s memory resources, which is squarely this library's mission; a `PoolAdaptor` if a non-pooling MR (`MallocMR`, the raw runtime MRs) proves common in hot paths.
