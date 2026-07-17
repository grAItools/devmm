"""`cupy.cuda.set_allocator` bridge, the mirror of rmm's
`rmm_cupy_allocator` (design §6): `install(mr)` points CuPy's allocator at a
devmm MR, so every CuPy allocation becomes a devmm `DeviceBuffer` surfaced
as an `UnownedMemory`-backed `MemoryPointer` — dropping the CuPy array
releases the buffer back through the MR chain.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any

from devmm._core.buffer import DeviceBuffer
from devmm._core.device import DeviceType
from devmm._core.memory_resource import DeviceMemoryResource
from devmm._runtimes.base import RuntimeUnavailableError
from devmm.integrations._support import (
    ForeignHandleStream,
    Installer,
    ensure_no_cycle,
    stream_handle_of,
)
from devmm.mrs.cuda import CupyAllocatorMemoryResource

__all__ = ["install"]


def _cupy_module() -> Any:
    """Seam for tests: the cupy module, imported on demand."""
    try:
        return importlib.import_module("cupy")
    except ImportError as exc:
        raise RuntimeUnavailableError(
            "cupy is not importable; install the devmm[cupy] extra to install "
            "a devmm memory resource into CuPy"
        ) from exc


def _make_allocator(cupy: Any, mr: DeviceMemoryResource) -> Callable[[int], Any]:
    def devmm_cupy_allocator(nbytes: int) -> Any:
        stream = ForeignHandleStream(mr.device, stream_handle_of(cupy.cuda.get_current_stream()))
        buffer = DeviceBuffer(nbytes, mr=mr, stream=stream)
        # device_id -1 lets CuPy infer the device from the pointer; a NULL
        # pointer carries nothing to infer from, so name the MR's device
        # then (the same distinction rmm_cupy_allocator makes).
        device_id = -1 if buffer.ptr else mr.device.index
        memory = cupy.cuda.UnownedMemory(buffer.ptr, buffer.nbytes, buffer, device_id)
        return cupy.cuda.MemoryPointer(memory, 0)

    return devmm_cupy_allocator


def install(mr: DeviceMemoryResource) -> Installer:
    """Point `cupy.cuda.set_allocator` at `mr` (design §6).

    An explicit call, never an import side effect. The returned `Installer`
    restores the previous CuPy allocator; arrays allocated while installed
    own their devmm buffers and stay valid (and freeable through `mr`) after
    uninstall. Installing an MR that itself consumes CuPy's allocator
    (`CupyAllocatorMemoryResource`) is refused: ``cupy.cuda.alloc`` calls
    the installed allocator, so the composition would recurse.
    """
    ensure_no_cycle(mr, CupyAllocatorMemoryResource, "CuPy")
    if mr.device.type is not DeviceType.CUDA:
        raise ValueError(
            f"CuPy allocates CUDA memory; the installed MR must live on a "
            f"cuda device, got {mr.device}"
        )
    cupy = _cupy_module()
    previous = cupy.cuda.get_allocator()
    cupy.cuda.set_allocator(_make_allocator(cupy, mr))

    def _restore() -> None:
        cupy.cuda.set_allocator(previous)

    return Installer("CuPy", _restore)
