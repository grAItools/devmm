"""Scripted CUDA test doubles (design §9): a `CudartApi` fake that records
every driver call in order and injects failures per entry point, plus rmm
doubles with the Python-layer MR/stream surface.
"""

from __future__ import annotations

from typing import Any

from devmm._runtimes.cuda import (
    CUDA_DEV_ATTR_MEMORY_POOLS_SUPPORTED,
    CUDA_SUCCESS,
    CudartApi,
)


class FakeCudartApi:
    """Records every libcudart call and injects failures (design §9).

    `fail[name] = status` makes entry point `name` return `status`; `calls`
    is the verbatim call log (`cudaGetErrorString` is a pure query and
    deliberately unrecorded). Fake handles are deterministic: pointers,
    streams and events come from disjoint ranges so a test can never
    mistake one for another.
    """

    def __init__(
        self,
        *,
        device_count: int = 2,
        current_device: int = 0,
        memory_pools_supported: bool = True,
    ) -> None:
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

    def _status(self, name: str) -> int:
        return self.fail.get(name, CUDA_SUCCESS)

    def _new_ptr(self, nbytes: int) -> int:
        ptr = self._next_ptr
        # 256-aligned, never-reused spans, mirroring the CUDA guarantee.
        self._next_ptr += -(-max(nbytes, 1) // 256) * 256
        self.live_allocations[ptr] = nbytes
        return ptr

    def cudaGetErrorString(self, status: int) -> str:
        return f"fake cudart error {status}"

    def cudaGetDeviceCount(self) -> tuple[int, int]:
        self.calls.append(("cudaGetDeviceCount",))
        return self._status("cudaGetDeviceCount"), self.device_count

    def cudaGetDevice(self) -> tuple[int, int]:
        self.calls.append(("cudaGetDevice",))
        return self._status("cudaGetDevice"), self.current_device

    def cudaSetDevice(self, index: int) -> int:
        self.calls.append(("cudaSetDevice", index))
        status = self._status("cudaSetDevice")
        if status == CUDA_SUCCESS:
            self.current_device = index
        return status

    def cudaDeviceGetAttribute(self, attribute: int, index: int) -> tuple[int, int]:
        self.calls.append(("cudaDeviceGetAttribute", attribute, index))
        value = 0
        if attribute == CUDA_DEV_ATTR_MEMORY_POOLS_SUPPORTED:
            value = int(self.memory_pools_supported)
        return self._status("cudaDeviceGetAttribute"), value

    def cudaMalloc(self, nbytes: int) -> tuple[int, int]:
        self.calls.append(("cudaMalloc", nbytes))
        status = self._status("cudaMalloc")
        if status != CUDA_SUCCESS:
            return status, 0
        return status, self._new_ptr(nbytes)

    def cudaFree(self, ptr: int) -> int:
        self.calls.append(("cudaFree", ptr))
        status = self._status("cudaFree")
        if status == CUDA_SUCCESS:
            self.live_allocations.pop(ptr, None)
        return status

    def cudaMallocAsync(self, nbytes: int, stream_handle: int) -> tuple[int, int]:
        self.calls.append(("cudaMallocAsync", nbytes, stream_handle))
        status = self._status("cudaMallocAsync")
        if status != CUDA_SUCCESS:
            return status, 0
        return status, self._new_ptr(nbytes)

    def cudaFreeAsync(self, ptr: int, stream_handle: int) -> int:
        self.calls.append(("cudaFreeAsync", ptr, stream_handle))
        status = self._status("cudaFreeAsync")
        if status == CUDA_SUCCESS:
            self.live_allocations.pop(ptr, None)
        return status

    def cudaStreamCreate(self) -> tuple[int, int]:
        self.calls.append(("cudaStreamCreate",))
        status = self._status("cudaStreamCreate")
        if status != CUDA_SUCCESS:
            return status, 0
        handle = self._next_stream
        self._next_stream += 1
        self.live_streams.add(handle)
        return status, handle

    def cudaStreamDestroy(self, handle: int) -> int:
        self.calls.append(("cudaStreamDestroy", handle))
        status = self._status("cudaStreamDestroy")
        if status == CUDA_SUCCESS:
            self.live_streams.discard(handle)
        return status

    def cudaStreamSynchronize(self, handle: int) -> int:
        self.calls.append(("cudaStreamSynchronize", handle))
        return self._status("cudaStreamSynchronize")

    def cudaMemcpyAsync(
        self, dst: int, src: int, nbytes: int, kind: int, stream_handle: int
    ) -> int:
        self.calls.append(("cudaMemcpyAsync", dst, src, nbytes, kind, stream_handle))
        return self._status("cudaMemcpyAsync")

    def cudaEventCreateWithFlags(self, flags: int) -> tuple[int, int]:
        self.calls.append(("cudaEventCreateWithFlags", flags))
        status = self._status("cudaEventCreateWithFlags")
        if status != CUDA_SUCCESS:
            return status, 0
        event = self._next_event
        self._next_event += 1
        self.live_events.add(event)
        return status, event

    def cudaEventRecord(self, event: int, stream_handle: int) -> int:
        self.calls.append(("cudaEventRecord", event, stream_handle))
        return self._status("cudaEventRecord")

    def cudaStreamWaitEvent(self, stream_handle: int, event: int, flags: int) -> int:
        self.calls.append(("cudaStreamWaitEvent", stream_handle, event, flags))
        return self._status("cudaStreamWaitEvent")

    def cudaEventDestroy(self, event: int) -> int:
        self.calls.append(("cudaEventDestroy", event))
        status = self._status("cudaEventDestroy")
        if status == CUDA_SUCCESS:
            self.live_events.discard(event)
        return status


# Structural conformance, verified by mypy --strict: the fake satisfies the
# injected-api protocol the runtime and raw MR are written against.
_PROTOCOL_CHECK: CudartApi = FakeCudartApi()


class StubNativeFunction:
    """One CDLL symbol: accepts the `restype`/`argtypes` prototype
    assignments `_LibcudartApi` performs at construction."""

    def __init__(self) -> None:
        self.restype: Any = None
        self.argtypes: Any = None


class AsynclessLibcudart:
    """CDLL-like double for a pre-11.2 libcudart: every symbol the api shim
    prototypes except the cudaMallocAsync/cudaFreeAsync pair, whose lookup
    raises AttributeError exactly as a real CDLL would."""

    _SYMBOLS = frozenset(
        {
            "cudaGetErrorString",
            "cudaGetDeviceCount",
            "cudaGetDevice",
            "cudaSetDevice",
            "cudaDeviceGetAttribute",
            "cudaMalloc",
            "cudaFree",
            "cudaStreamCreate",
            "cudaStreamDestroy",
            "cudaStreamSynchronize",
            "cudaMemcpyAsync",
            "cudaEventCreateWithFlags",
            "cudaEventRecord",
            "cudaStreamWaitEvent",
            "cudaEventDestroy",
        }
    )

    def __init__(self) -> None:
        self._functions: dict[str, StubNativeFunction] = {}

    def __getattr__(self, name: str) -> StubNativeFunction:
        if name not in self._SYMBOLS:
            raise AttributeError(name)
        return self._functions.setdefault(name, StubNativeFunction())


class FakeRmmStream:
    """Stands in for `rmm.pylibrmm.stream.Stream`: remembers what it wrapped."""

    def __init__(self, obj: object) -> None:
        self.wrapped = obj


class FakeRmmMemoryResource:
    """An rmm `DeviceMemoryResource` double with the Python-layer signature
    (`allocate(nbytes, stream)` / `deallocate(ptr, nbytes, stream)`)."""

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
