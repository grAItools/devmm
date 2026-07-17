"""Scripted GPU test doubles (design §9): a `GpuApi` fake bound to a
`GpuPlatform` that records every native call in order — under the platform's
own symbol spellings — and injects failures per entry point, plus rmm doubles
with the Python-layer MR/stream surface rmm and hipMM share, and the harness
table the CUDA/ROCm-parametrized suites run over (design §4.2).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from types import ModuleType
from typing import Any

from devmm._core.device import Device, DeviceType
from devmm._core.memory_resource import DeviceMemoryResource
from devmm._core.stream import DEFAULT, LEGACY_DEFAULT, PER_THREAD_DEFAULT, StreamSentinel
from devmm._runtimes import _discovery
from devmm._runtimes import cuda as cuda_module
from devmm._runtimes import rocm as rocm_module
from devmm._runtimes._gpulib import (
    GPU_SUCCESS,
    GpuApi,
    GpuError,
    GpuPlatform,
    GpuRuntime,
    GpuStream,
    GpuSymbols,
)
from devmm.mrs import cuda as mrs_cuda
from devmm.mrs import rocm as mrs_rocm


class FakeGpuApi:
    """Records every native call and injects failures (design §9).

    Call-log entries and `fail` keys use the bound platform's native symbol
    spellings (`platform.symbols`), so the same suite reads as cudaMalloc
    calls under the CUDA harness and hipMalloc calls under the ROCm one.
    (`get_error_string` is a pure query and deliberately unrecorded.) Fake
    handles are deterministic: pointers, streams and events come from
    disjoint ranges so a test can never mistake one for another.
    """

    def __init__(
        self,
        platform: GpuPlatform,
        *,
        device_count: int = 2,
        current_device: int = 0,
        memory_pools_supported: bool = True,
    ) -> None:
        self.platform = platform
        self.calls: list[tuple[Any, ...]] = []
        self.fail: dict[str, int] = {}
        self.device_count = device_count
        self.current_device = current_device
        self.memory_pools_supported = memory_pools_supported
        self.live_allocations: dict[int, int] = {}
        self.live_streams: set[int] = set()
        self.live_events: set[int] = set()
        self._next_ptr = 0x10000
        self._next_stream = 0x51000
        self._next_event = 0xE1000

    def _status(self, symbol: str) -> int:
        return self.fail.get(symbol, GPU_SUCCESS)

    def _new_ptr(self, nbytes: int) -> int:
        ptr = self._next_ptr
        # 256-aligned, never-reused spans, mirroring the CUDA/HIP guarantee.
        self._next_ptr += -(-max(nbytes, 1) // 256) * 256
        self.live_allocations[ptr] = nbytes
        return ptr

    def get_error_string(self, status: int) -> str:
        return f"fake {self.platform.name} error {status}"

    def get_device_count(self) -> tuple[int, int]:
        symbol = self.platform.symbols.get_device_count
        self.calls.append((symbol,))
        return self._status(symbol), self.device_count

    def get_device(self) -> tuple[int, int]:
        symbol = self.platform.symbols.get_device
        self.calls.append((symbol,))
        return self._status(symbol), self.current_device

    def set_device(self, index: int) -> int:
        symbol = self.platform.symbols.set_device
        self.calls.append((symbol, index))
        status = self._status(symbol)
        if status == GPU_SUCCESS:
            self.current_device = index
        return status

    def get_device_attribute(self, attribute: int, index: int) -> tuple[int, int]:
        symbol = self.platform.symbols.get_device_attribute
        self.calls.append((symbol, attribute, index))
        value = 0
        if attribute == self.platform.memory_pools_attribute:
            value = int(self.memory_pools_supported)
        return self._status(symbol), value

    def malloc(self, nbytes: int) -> tuple[int, int]:
        symbol = self.platform.symbols.malloc
        self.calls.append((symbol, nbytes))
        status = self._status(symbol)
        if status != GPU_SUCCESS:
            return status, 0
        return status, self._new_ptr(nbytes)

    def free(self, ptr: int) -> int:
        symbol = self.platform.symbols.free
        self.calls.append((symbol, ptr))
        status = self._status(symbol)
        if status == GPU_SUCCESS:
            self.live_allocations.pop(ptr, None)
        return status

    def malloc_async(self, nbytes: int, stream_handle: int) -> tuple[int, int]:
        symbol = self.platform.symbols.malloc_async
        self.calls.append((symbol, nbytes, stream_handle))
        status = self._status(symbol)
        if status != GPU_SUCCESS:
            return status, 0
        return status, self._new_ptr(nbytes)

    def free_async(self, ptr: int, stream_handle: int) -> int:
        symbol = self.platform.symbols.free_async
        self.calls.append((symbol, ptr, stream_handle))
        status = self._status(symbol)
        if status == GPU_SUCCESS:
            self.live_allocations.pop(ptr, None)
        return status

    def stream_create(self) -> tuple[int, int]:
        symbol = self.platform.symbols.stream_create
        self.calls.append((symbol,))
        status = self._status(symbol)
        if status != GPU_SUCCESS:
            return status, 0
        handle = self._next_stream
        self._next_stream += 1
        self.live_streams.add(handle)
        return status, handle

    def stream_destroy(self, handle: int) -> int:
        symbol = self.platform.symbols.stream_destroy
        self.calls.append((symbol, handle))
        status = self._status(symbol)
        if status == GPU_SUCCESS:
            self.live_streams.discard(handle)
        return status

    def stream_synchronize(self, handle: int) -> int:
        symbol = self.platform.symbols.stream_synchronize
        self.calls.append((symbol, handle))
        return self._status(symbol)

    def memcpy_async(self, dst: int, src: int, nbytes: int, kind: int, stream_handle: int) -> int:
        symbol = self.platform.symbols.memcpy_async
        self.calls.append((symbol, dst, src, nbytes, kind, stream_handle))
        return self._status(symbol)

    def event_create_with_flags(self, flags: int) -> tuple[int, int]:
        symbol = self.platform.symbols.event_create_with_flags
        self.calls.append((symbol, flags))
        status = self._status(symbol)
        if status != GPU_SUCCESS:
            return status, 0
        event = self._next_event
        self._next_event += 1
        self.live_events.add(event)
        return status, event

    def event_record(self, event: int, stream_handle: int) -> int:
        symbol = self.platform.symbols.event_record
        self.calls.append((symbol, event, stream_handle))
        return self._status(symbol)

    def stream_wait_event(self, stream_handle: int, event: int, flags: int) -> int:
        symbol = self.platform.symbols.stream_wait_event
        self.calls.append((symbol, stream_handle, event, flags))
        return self._status(symbol)

    def event_destroy(self, event: int) -> int:
        symbol = self.platform.symbols.event_destroy
        self.calls.append((symbol, event))
        status = self._status(symbol)
        if status == GPU_SUCCESS:
            self.live_events.discard(event)
        return status


# Structural conformance, verified by mypy --strict: the fake satisfies the
# injected-api protocol the runtimes and raw MRs are written against.
_PROTOCOL_CHECK: GpuApi = FakeGpuApi(cuda_module.CUDA_PLATFORM)


class StubNativeFunction:
    """One CDLL symbol: accepts the `restype`/`argtypes` prototype
    assignments `NativeGpuApi` performs at construction."""

    def __init__(self) -> None:
        self.restype: Any = None
        self.argtypes: Any = None


class AsynclessNativeLib:
    """CDLL-like double for a runtime library predating the async allocation
    family (CUDA < 11.2, ROCm < 5.2): every symbol in the platform's table
    except the malloc_async/free_async pair, whose lookup raises
    AttributeError exactly as a real CDLL would."""

    def __init__(self, platform: GpuPlatform) -> None:
        symbols = platform.symbols
        names = {getattr(symbols, field.name) for field in dataclasses.fields(GpuSymbols)}
        self._symbols = names - {symbols.malloc_async, symbols.free_async}
        self._functions: dict[str, StubNativeFunction] = {}

    def __getattr__(self, name: str) -> StubNativeFunction:
        if name not in self._symbols:
            raise AttributeError(name)
        return self._functions.setdefault(name, StubNativeFunction())


class FakeRmmStream:
    """Stands in for `rmm.pylibrmm.stream.Stream`: remembers what it wrapped."""

    def __init__(self, obj: object) -> None:
        self.wrapped = obj


class FakeRmmMemoryResource:
    """An rmm/hipMM `DeviceMemoryResource` double with the Python-layer
    signature (`allocate(nbytes, stream)` / `deallocate(ptr, nbytes, stream)`)."""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []
        self._next_ptr = 0x200000

    def allocate(self, nbytes: int, stream: Any) -> int:
        self.calls.append(("allocate", nbytes, stream))
        ptr = self._next_ptr
        self._next_ptr += -(-max(nbytes, 1) // 256) * 256
        return ptr

    def deallocate(self, ptr: int, nbytes: int, stream: Any) -> None:
        self.calls.append(("deallocate", ptr, nbytes, stream))


@dataclass(frozen=True)
class GpuHarness:
    """One platform binding of the shared CUDA/ROCm suite (design §4.2):
    everything that legitimately differs between the two — modules, classes,
    devices, driver-library spellings and expected magic handles."""

    name: str
    platform: GpuPlatform
    runtime_module: ModuleType
    mrs_module: ModuleType
    runtime_cls: type[GpuRuntime]
    stream_cls: type[GpuStream]
    error_cls: type[GpuError]
    raw_mr_cls: type[DeviceMemoryResource]
    rmm_mr_cls: type[DeviceMemoryResource]
    device0: Device
    device1: Device
    foreign_device_type: DeviceType
    driver_libraries: tuple[str, ...]
    expected_sentinels: tuple[tuple[StreamSentinel, int], ...]
    library_match: str
    # The `rmm.mr` marker class this platform's build of the module ships
    # (design §4.2): RMM is Cuda*-named, hipMM is Hip*-named.
    rmm_marker: str

    def api(self, **kwargs: int | bool) -> FakeGpuApi:
        return FakeGpuApi(self.platform, **kwargs)  # type: ignore[arg-type]


HARNESSES: dict[str, GpuHarness] = {
    "cuda": GpuHarness(
        name="cuda",
        platform=cuda_module.CUDA_PLATFORM,
        runtime_module=cuda_module,
        mrs_module=mrs_cuda,
        runtime_cls=cuda_module.CudaRuntime,
        stream_cls=cuda_module.CudaStream,
        error_cls=cuda_module.CudaError,
        raw_mr_cls=mrs_cuda.CudaRuntimeMemoryResource,
        rmm_mr_cls=mrs_cuda.RmmMemoryResource,
        device0=Device.from_string("cuda:0"),
        device1=Device.from_string("cuda:1"),
        foreign_device_type=DeviceType.ROCM,
        driver_libraries=_discovery._CUDA_DRIVER_LIBRARIES,
        # cudaStreamDefault / cudaStreamLegacy / cudaStreamPerThread.
        expected_sentinels=((DEFAULT, 0), (LEGACY_DEFAULT, 1), (PER_THREAD_DEFAULT, 2)),
        library_match="libcudart",
        rmm_marker="CudaMemoryResource",
    ),
    "rocm": GpuHarness(
        name="rocm",
        platform=rocm_module.HIP_PLATFORM,
        runtime_module=rocm_module,
        mrs_module=mrs_rocm,
        runtime_cls=rocm_module.HipRuntime,
        stream_cls=rocm_module.HipStream,
        error_cls=rocm_module.HipError,
        raw_mr_cls=mrs_rocm.HipRuntimeMemoryResource,
        rmm_mr_cls=mrs_rocm.HipmmMemoryResource,
        device0=Device.from_string("rocm:0"),
        device1=Device.from_string("rocm:1"),
        foreign_device_type=DeviceType.CUDA,
        driver_libraries=_discovery._ROCM_DRIVER_LIBRARIES,
        # HIP's null stream doubles as its legacy default (the DLPack table
        # in `_dlpack/export.py` maps ROCm's legacy default to 0 too);
        # hipStreamPerThread is (hipStream_t)2.
        expected_sentinels=((DEFAULT, 0), (LEGACY_DEFAULT, 0), (PER_THREAD_DEFAULT, 2)),
        library_match="libamdhip64",
        rmm_marker="HipMemoryResource",
    ),
}
