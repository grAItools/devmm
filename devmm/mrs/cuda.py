"""CUDA memory resources (design §5.2).

`RmmMemoryResource` wraps any `rmm.mr.DeviceMemoryResource`, translating
streams through the CUDA stream protocol. `CudaRuntimeMemoryResource` is
plain cudaMalloc/cudaFree — cudaMallocAsync/cudaFreeAsync when the driver
supports memory pools — via ctypes on libcudart with no third-party
dependency. Both are thin CUDA bindings of the shared GPU shim
(`devmm._runtimes._gpulib`, design §4.2), unit-testable without hardware
over an injected `GpuApi` (design §9). `CupyAllocatorMemoryResource` wraps
any CuPy-compatible allocator, bridging teams whose GPU memory budget is
already governed by a CuPy pool.
"""

from __future__ import annotations

import importlib
import threading
from collections.abc import Callable
from typing import Any

from devmm._core.device import Device, DeviceType
from devmm._core.memory_resource import DeviceMemoryResource
from devmm._core.stream import Stream
from devmm._runtimes._gpulib import (
    GpuRuntimeMemoryResource,
    RmmLikeMemoryResource,
    rmm_stream_class,
)
from devmm._runtimes.base import RuntimeUnavailableError
from devmm._runtimes.cuda import CUDA_PLATFORM

__all__ = [
    "CudaRuntimeMemoryResource",
    "CupyAllocatorMemoryResource",
    "RmmMemoryResource",
]

_CUDA_DEVICE = Device(DeviceType.CUDA, 0)


def _rmm_stream_class() -> Any:
    """Seam for tests: rmm's Stream class, imported on demand (design §5.2)."""
    return rmm_stream_class(
        "rmm is not importable; install the devmm[cuda] extra to use RmmMemoryResource"
    )


def _cupy_module() -> Any:
    """Seam for tests: the cupy module, imported on demand (design §5.2)."""
    try:
        return importlib.import_module("cupy")
    except ImportError as exc:
        raise RuntimeUnavailableError(
            "cupy is not importable; install the devmm[cupy] extra to use "
            "CupyAllocatorMemoryResource"
        ) from exc


class CudaRuntimeMemoryResource(GpuRuntimeMemoryResource):
    """The "just cudaMalloc" MR (design §5.2), no third-party dependency;
    semantics in `GpuRuntimeMemoryResource`."""

    platform = CUDA_PLATFORM


class RmmMemoryResource(RmmLikeMemoryResource):
    """The flagship wrapper over any `rmm.mr.DeviceMemoryResource`
    (design §5.2); semantics in `RmmLikeMemoryResource`."""

    platform = CUDA_PLATFORM

    def _translated_stream(self, stream: Stream) -> Any:
        return _rmm_stream_class()(stream)


class CupyAllocatorMemoryResource(DeviceMemoryResource):
    """Wrapper over any CuPy-compatible allocator (design §5.2): a callable
    ``f(nbytes) -> cupy.cuda.MemoryPointer`` — a ``MemoryPool().malloc``, a
    user allocator, or the default None meaning ``cupy.cuda.alloc`` (CuPy's
    current allocator, usually the default pool).

    CuPy pools key cached blocks by the thread-local current stream, so
    every allocation runs with the MR's device and the caller's stream
    current. The returned `MemoryPointer` is stashed until `deallocate`
    drops it — CuPy frees/returns-to-pool on refcount zero under its own
    stream-safety rules, hence `stream_ordered=True`. CuPy allocation
    failures (`OutOfMemoryError` subclasses `MemoryError`) propagate
    untouched.

    The mirror direction — pointing CuPy's allocator at a devmm MR — is
    `devmm.integrations.cupy.install` (design §6); composing the two is
    refused there because ``cupy.cuda.alloc`` calls the installed allocator.
    """

    def __init__(
        self,
        allocator: Callable[[int], Any] | None = None,
        device: Device = _CUDA_DEVICE,
    ) -> None:
        if device.type is not DeviceType.CUDA:
            raise ValueError(f"{type(self).__name__} requires a cuda device, got {device}")
        self.device = device
        self._allocator = allocator
        self._lock = threading.Lock()
        # ptr -> (nbytes, MemoryPointer): the stashed MemoryPointer is what
        # keeps CuPy's block alive; dropping the entry is the free.
        self._live: dict[int, tuple[int, Any]] = {}

    def allocate(self, nbytes: int, stream: Stream) -> int:
        if nbytes < 0:
            raise ValueError(f"cannot allocate a negative size ({nbytes} bytes)")
        cupy = _cupy_module()
        allocator = cupy.cuda.alloc if self._allocator is None else self._allocator
        # max(nbytes, 1): a zero-byte pool request yields a NULL
        # MemoryPointer, which could be neither tracked nor freed; one byte
        # buys a unique, freeable pointer.
        with cupy.cuda.Device(self.device.index), cupy.cuda.ExternalStream(stream.handle):
            memory_pointer = allocator(max(nbytes, 1))
        ptr = int(memory_pointer.ptr)
        with self._lock:
            self._live[ptr] = (nbytes, memory_pointer)
        return ptr

    def deallocate(self, ptr: int, nbytes: int, stream: Stream) -> None:
        with self._lock:
            entry = self._live.get(ptr)
            if entry is None:
                raise ValueError(
                    f"pointer {ptr:#x} is not a live allocation of {self!r} "
                    "(double-free, or a pointer this MR never returned)"
                )
            recorded, _ = entry
            if nbytes != recorded:
                # The allocation stays live: a size-mismatched free is caller
                # confusion, and releasing anyway would turn it into a
                # use-after-free elsewhere.
                raise ValueError(
                    f"size mismatch freeing pointer {ptr:#x} in {self!r}: "
                    f"allocated {recorded} bytes, deallocate got {nbytes}"
                )
            del self._live[ptr]

    @property
    def stream_ordered(self) -> bool:
        return True

    def guaranteed_alignment(self) -> int:
        # Every CuPy-compatible allocator hands out CUDA device memory, so
        # allocations inherit the platform's 256-byte guarantee (CuPy's own
        # pool rounds further, to 512).
        return 256

    def _debug_live_count(self) -> int:
        """Testing hook: allocations currently tracked (0 == tables empty)."""
        with self._lock:
            return len(self._live)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(allocator={self._allocator!r}, device={self.device})"
