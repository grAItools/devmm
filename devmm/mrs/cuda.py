"""CUDA memory resources (design §5.2).

`RmmMemoryResource` wraps any `rmm.mr.DeviceMemoryResource`, translating
streams through the CUDA stream protocol. `CudaRuntimeMemoryResource` is
plain cudaMalloc/cudaFree — cudaMallocAsync/cudaFreeAsync when the driver
supports memory pools — via ctypes on libcudart with no third-party
dependency; its control flow runs over an injected `CudartApi` (design §9)
so it is unit-testable without hardware.
"""

from __future__ import annotations

import importlib
import threading
from typing import Any, Literal, Protocol

from devmm._core.device import Device, DeviceType
from devmm._core.memory_resource import DeviceMemoryResource
from devmm._core.stream import Stream
from devmm._runtimes.base import RuntimeUnavailableError
from devmm._runtimes.cuda import (
    CUDA_DEV_ATTR_MEMORY_POOLS_SUPPORTED,
    CUDA_SUCCESS,
    CudartApi,
    _check,
    _default_api,
    _device_activation,
)

__all__ = [
    "CudaRuntimeMemoryResource",
    "RmmMemoryResource",
]


def _check_cuda_device(device: Device, mr_name: str) -> None:
    if device.type is not DeviceType.CUDA:
        raise ValueError(f"{mr_name} requires a cuda device, got {device}")


class CudaRuntimeMemoryResource(DeviceMemoryResource):
    """The "just cudaMalloc" MR (design §5.2), no third-party dependency.

    `async_alloc="auto"` probes cudaDevAttrMemoryPoolsSupported once and
    picks the cudaMallocAsync/cudaFreeAsync family iff the driver supports
    memory pools (a failed probe means "not supported": older drivers
    predate the attribute); `True`/`False` force the async/sync family
    without probing. `stream_ordered` is True only on the async path — the
    sync path relies on cudaFree's implicit synchronization for safety.
    Both families return pointers aligned to at least 256 bytes.
    """

    def __init__(
        self,
        device: Device,
        *,
        async_alloc: bool | Literal["auto"] = "auto",
        api: CudartApi | None = None,
    ) -> None:
        _check_cuda_device(device, type(self).__name__)
        self.device = device
        self._api = _default_api() if api is None else api
        if async_alloc == "auto":
            status, supported = self._api.cudaDeviceGetAttribute(
                CUDA_DEV_ATTR_MEMORY_POOLS_SUPPORTED, device.index
            )
            self._async = status == CUDA_SUCCESS and bool(supported)
        elif async_alloc is True or async_alloc is False:
            self._async = async_alloc
        else:
            raise ValueError(f"async_alloc must be 'auto', True or False, got {async_alloc!r}")
        self._lock = threading.Lock()
        # ptr -> nbytes: deallocate validates the pointer and size before any
        # driver call, so misuse surfaces as ValueError, never memory
        # corruption.
        self._live: dict[int, int] = {}

    def allocate(self, nbytes: int, stream: Stream) -> int:
        if nbytes < 0:
            raise ValueError(f"cannot allocate a negative size ({nbytes} bytes)")
        # max(nbytes, 1): cudaMalloc(0) returns NULL, which could be neither
        # tracked nor freed; one byte buys a unique, freeable pointer.
        request = max(nbytes, 1)
        with _device_activation(self._api, self.device):
            if self._async:
                status, ptr = self._api.cudaMallocAsync(request, stream.handle)
            else:
                status, ptr = self._api.cudaMalloc(request)
        if status != CUDA_SUCCESS:
            raise MemoryError(
                f"failed to allocate {nbytes} bytes on {self.device} in {self!r}: "
                f"{self._api.cudaGetErrorString(status)}"
            )
        with self._lock:
            self._live[ptr] = nbytes
        return ptr

    def deallocate(self, ptr: int, nbytes: int, stream: Stream) -> None:
        with self._lock:
            recorded = self._live.get(ptr)
            if recorded is None:
                raise ValueError(
                    f"pointer {ptr:#x} is not a live allocation of {self!r} "
                    "(double-free, or a pointer this MR never returned)"
                )
            if nbytes != recorded:
                # The allocation stays live: a size-mismatched free is caller
                # confusion, and releasing anyway would turn it into a
                # use-after-free elsewhere.
                raise ValueError(
                    f"size mismatch freeing pointer {ptr:#x} in {self!r}: "
                    f"allocated {recorded} bytes, deallocate got {nbytes}"
                )
            del self._live[ptr]
        with _device_activation(self._api, self.device):
            if self._async:
                status = self._api.cudaFreeAsync(ptr, stream.handle)
                function = "cudaFreeAsync"
            else:
                status = self._api.cudaFree(ptr)
                function = "cudaFree"
        _check(self._api, function, status)

    @property
    def stream_ordered(self) -> bool:
        return self._async

    def guaranteed_alignment(self) -> int:
        # CUDA guarantees at least 256-byte alignment from both families.
        return 256

    def _debug_live_count(self) -> int:
        """Testing hook: allocations currently tracked (0 == tables empty)."""
        with self._lock:
            return len(self._live)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(device={self.device}, async_alloc={self._async})"


def _rmm_stream_class() -> Any:
    """rmm's Stream class, imported on demand (seam for hardware-free tests).

    rmm's Stream constructor accepts any object exposing `__cuda_stream__`
    — which every devmm `Stream` does — so wrapping the devmm stream in it
    is the whole translation (design §5.2).
    """
    try:
        return importlib.import_module("rmm.pylibrmm.stream").Stream
    except ImportError as exc:
        raise RuntimeUnavailableError(
            "rmm is not importable; install the devmm[cuda] extra to use RmmMemoryResource"
        ) from exc


class RmmResourceLike(Protocol):
    """The rmm `DeviceMemoryResource` Python surface `RmmMemoryResource`
    forwards to — a Protocol so the zero-dependency core never imports rmm
    (design §8)."""

    def allocate(self, nbytes: int, stream: Any) -> int: ...

    def deallocate(self, ptr: int, nbytes: int, stream: Any) -> None: ...


class RmmMemoryResource(DeviceMemoryResource):
    """The flagship wrapper over any `rmm.mr.DeviceMemoryResource`
    (design §5.2).

    `inner` is a strong reference by design — the rmm lifetime lesson
    (design §3.3): the wrapped resource must outlive every allocation made
    through it. Streams are translated via the CUDA stream protocol
    (`_rmm_stream_class`); allocation failures raised by rmm (already
    `MemoryError`s) propagate untouched. rmm resources allocate
    stream-ordered, 256-byte-aligned memory. The caller pairs `inner` with
    the device it was created for — rmm resources do not expose it.
    """

    inner: RmmResourceLike

    def __init__(self, inner: RmmResourceLike, device: Device) -> None:
        _check_cuda_device(device, type(self).__name__)
        self.inner = inner
        self.device = device
        self._lock = threading.Lock()
        # ptr -> nbytes, for the same misuse detection the sibling MRs give.
        self._live: dict[int, int] = {}

    def allocate(self, nbytes: int, stream: Stream) -> int:
        if nbytes < 0:
            raise ValueError(f"cannot allocate a negative size ({nbytes} bytes)")
        # max(nbytes, 1): rmm forwards 0 to allocators that may return NULL
        # or a shared address; one byte buys a unique, freeable pointer.
        ptr = int(self.inner.allocate(max(nbytes, 1), _rmm_stream_class()(stream)))
        with self._lock:
            self._live[ptr] = nbytes
        return ptr

    def deallocate(self, ptr: int, nbytes: int, stream: Stream) -> None:
        with self._lock:
            recorded = self._live.get(ptr)
            if recorded is None:
                raise ValueError(
                    f"pointer {ptr:#x} is not a live allocation of {self!r} "
                    "(double-free, or a pointer this MR never returned)"
                )
            if nbytes != recorded:
                raise ValueError(
                    f"size mismatch freeing pointer {ptr:#x} in {self!r}: "
                    f"allocated {recorded} bytes, deallocate got {nbytes}"
                )
            del self._live[ptr]
        self.inner.deallocate(ptr, max(nbytes, 1), _rmm_stream_class()(stream))

    @property
    def stream_ordered(self) -> bool:
        return True

    def guaranteed_alignment(self) -> int:
        # rmm's contract: allocations are aligned to at least 256 bytes.
        return 256

    def _debug_live_count(self) -> int:
        """Testing hook: allocations currently tracked (0 == tables empty)."""
        with self._lock:
            return len(self._live)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(inner={self.inner!r}, device={self.device})"
