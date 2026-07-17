# devmm API reference

The public surface re-exported from `devmm` (design §2), with executable
examples: every Python block below is a doctest session run by
`tests/test_docs.py`, so this document cannot drift from the library.
Examples share one namespace, top to bottom.

```python
>>> import numpy as np
>>> import devmm

```

## Devices

`Device` is a frozen value object; `DeviceType` values are exactly the DLPack
`DLDeviceType` codes, so `__dlpack_device__` needs no translation table.
Every buffer, stream and memory resource carries an explicit `Device` — there
is no ambient "current device" (design §3.1).

```python
>>> cpu = devmm.Device.from_string("cpu")
>>> cpu
Device(type=<DeviceType.CPU: 1>, index=0)
>>> devmm.Device.from_string("cuda:1").__dlpack_device__()
(2, 1)
>>> int(devmm.DeviceType.ROCM)
10

```

## Streams

Streams are opaque handles plus ordering primitives (`synchronize`,
`wait_raw`); devmm never launches kernels. Obtain them from the device's
runtime; the sentinels `DEFAULT`, `LEGACY_DEFAULT` and `PER_THREAD_DEFAULT`
map to the platform's magic handles inside each runtime (design §3.2).

```python
>>> stream = devmm.runtime_for(cpu).default_stream(cpu)
>>> stream.handle  # the CPU runtime's single no-op stream
0
>>> stream.synchronize()  # no-op on CPU; real ordering on CUDA/ROCm
>>> devmm.DEFAULT is devmm.LEGACY_DEFAULT
False

```

## Layout policies

A `LayoutPolicy` maps `(shape, dtype, device)` to a concrete, resolved
`Layout` — a permutation-derived stride set (in elements, DLPack convention)
plus padding and alignment requirements. Shipped policies: `RowMajor`,
`ColMajor`, `Permuted`, `Aligned`, `DeviceOptimal` (design §3.6).

```python
>>> f32 = devmm.DType.from_string("float32")
>>> devmm.RowMajor()((2, 3), f32, cpu).strides
(3, 1)
>>> devmm.ColMajor()((2, 3), f32, cpu).strides
(1, 2)
>>> aligned = devmm.Aligned(devmm.RowMajor(), unit_stride_alignment=128, base_alignment=256)
>>> layout = aligned((4, 3), f32, cpu)
>>> layout.strides, layout.required_nbytes % 256
((32, 1), 0)
>>> layout.policy is aligned  # layouts carry provenance
True

```

## Allocation: `empty`, `empty_like`, `Tensor`

`empty` is the whole user story in one call: policy → `Layout` →
`mr.allocate` → `DeviceBuffer` → `Tensor`. The `Tensor` is not an array — no
arithmetic, no indexing — just a typed, shaped, DLPack-exportable view over
an owning, stream-ordered `DeviceBuffer` (design §3.5, §3.8).

```python
>>> t = devmm.empty((2, 3), "float32", layout=devmm.RowMajor())
>>> t.shape, t.strides, t.device
((2, 3), (3, 1), Device(type=<DeviceType.CPU: 1>, index=0))
>>> like = devmm.empty_like(np.zeros((2, 2), dtype=np.int32))
>>> like.shape, like.dtype == devmm.DType.from_string("int32")
((2, 2), True)

```

`DeviceBuffer.free()` is idempotent and stream-ordered on the allocation
stream by default; buffers are context managers for deterministic release,
with a `weakref.finalize` safety net for the rest.

```python
>>> t.buffer.free()
>>> t.buffer.free()  # idempotent
>>> t.buffer.closed
True

```

## DLPack export

Tensors implement `__dlpack__`/`__dlpack_device__` with DLPack 1.1 version
negotiation (versioned and legacy capsules), the `stream=` consumer handoff,
and `BufferError` for every refusal per the protocol: cross-device
`dl_device`, `copy=True`, freed buffers, read-only tensors facing legacy-only
consumers (design §7).

```python
>>> producer = devmm.empty((2, 2), "float32", layout=devmm.RowMajor())
>>> producer.__dlpack_device__()
(1, 0)
>>> consumed = np.from_dlpack(producer)  # negotiates and consumes the capsule
>>> consumed.shape
(2, 2)
>>> producer.__dlpack__(copy=True)
Traceback (most recent call last):
    ...
BufferError: copy=True is not supported; devmm exports are always zero-copy

```

## Memory resources and adaptors

`DeviceMemoryResource` is deliberately isomorphic to rmm's:
`allocate(nbytes, stream) -> int` and `deallocate(ptr, nbytes, stream)`, with
capability probes `stream_ordered`, `guaranteed_alignment()` and
`available_memory()` (design §3.3). Concrete MRs live in `devmm.mrs.*`:

- `mrs.cpu` — `BytearrayMemoryResource`, `MallocMemoryResource`,
  `NumpyHandlerMemoryResource` (experimental, NEP-49).
- `mrs.cuda` — `CudaRuntimeMemoryResource`, `RmmMemoryResource`,
  `CupyAllocatorMemoryResource`.
- `mrs.rocm` — `HipRuntimeMemoryResource`, `HipmmMemoryResource`.

Adaptors compose over any MR and hold their upstream strongly, so the full
allocator chain outlives every allocation:

```python
>>> from devmm.mrs.cpu import MallocMemoryResource
>>> stats = devmm.StatisticsAdaptor(MallocMemoryResource())
>>> ptr = stats.allocate(64, stream)
>>> stats.current_bytes, stats.total_bytes
(64, 64)
>>> stats.deallocate(ptr, 64, stream)
>>> stats.current_bytes
0
>>> limited = devmm.LimitingAdaptor(MallocMemoryResource(), limit_bytes=64)
>>> limited.allocate(65, stream)
Traceback (most recent call last):
    ...
MemoryError: ...

```

`LoggingAdaptor` emits structured records on the `devmm.mr` logger, and
`CallbackMemoryResource(alloc_fn, dealloc_fn, device)` is the pure-Python
escape hatch. Pooling is out of scope — configure rmm/hipMM/CuPy pools
underneath instead.

## The current-MR registry

A per-device registry of strong references with a `contextvars`-scoped
override, so it composes with threads and async tasks. The lazy default asks
the device's runtime (CPU → `MallocMemoryResource`; CUDA → rmm when
importable, else the raw runtime MR; likewise ROCm) (design §3.4).

```python
>>> devmm.get_current_memory_resource(cpu)
MallocMemoryResource(device=cpu:0, alignment=64)
>>> with devmm.using_memory_resource(stats) as active:
...     devmm.get_current_memory_resource(cpu) is active
True
>>> devmm.get_current_memory_resource(cpu) is stats  # restored on exit
False

```

`set_current_memory_resource(mr)` installs a process-wide current MR keyed by
`mr.device`.

## Runtimes

One `DeviceRuntime` per platform: device enumeration, stream factories,
`memcpy`, native device activation, and the default-MR policy.
`runtime_names()` probes without importing anything heavyweight;
`runtime_for` loads on demand; `DEVMM_RUNTIME=cpu|cuda|rocm` overrides
discovery, and third parties register via the `devmm.runtimes` entry-point
group (design §4).

```python
>>> "cpu" in devmm.runtime_names()
True
>>> devmm.runtime_for("cpu").name
'cpu'
>>> devmm.runtime_for(cpu) is devmm.runtime_for(devmm.DeviceType.CPU)
True

```

## Integrations

For every ecosystem allocator there are two arrows (design §6): *consume* its
allocator as a devmm MR (`devmm.mrs.*`), or *provide* a devmm MR to it
(`devmm.integrations.*`). Provide arrows mutate third-party global state, so
they are always explicit `install(mr)` calls returning an `Installer` whose
`uninstall()` (or context exit) restores the prior state — never import side
effects. Installing both arrows for the same library is a cycle and raises.

- `integrations.numpy.install(mr)` — NEP-49 data-memory handler.
- `integrations.cupy.install(mr)` — `cupy.cuda.set_allocator` bridge.
- `integrations.numba` — `DevmmEMMPlugin` external memory manager.
- `integrations.rmm.install(mr)` — rmm per-device resource bridge.

The NumPy arrow runs anywhere NumPy does:

```python
>>> import devmm.integrations.numpy
>>> numpy_stats = devmm.StatisticsAdaptor(MallocMemoryResource())
>>> with devmm.integrations.numpy.install(numpy_stats):
...     arr = np.ones(1000)
>>> numpy_stats.total_bytes >= 8000  # every NumPy allocation was accounted
True

```

## Testing: conformance suites and the recording MR

`devmm.testing` ships the tools the suite itself is built on (design §9), so
third-party MR and runtime authors get the same guarantees:

- `mr_conformance(mr_factory)` runs the memory-resource contract suite
  against fresh MRs from `mr_factory` (byte-exact writes/reads, no aliasing,
  alignment honesty, `_debug_live_count()` bookkeeping, misuse detection,
  zero-byte and negative-size contracts). Defaults dereference pointers on
  the host; pass `write=`/`read=` backed by your runtime's `memcpy` (and a
  `stream_factory`) for device memory.
- `dlpack_conformance(device)` runs the DLPack producer contract suite for
  tensors on `device` — capsule flavours and version stamping, struct-field
  fidelity, refusal `BufferError`s, and holder lifetime — without
  dereferencing device memory.
- `RecordingMemoryResource` is an allocator double with deterministic fake
  pointers, a verbatim call log, and misuse detection.

```python
>>> import devmm.testing
>>> from devmm.mrs.cpu import BytearrayMemoryResource
>>> devmm.testing.mr_conformance(BytearrayMemoryResource)  # raises on violation
>>> devmm.testing.dlpack_conformance(cpu)
>>> recording = devmm.testing.RecordingMemoryResource()
>>> buf = devmm.DeviceBuffer(64, mr=recording, stream=stream)
>>> recording.live
{256: 64}
>>> buf.free()
>>> recording.live
{}

```
