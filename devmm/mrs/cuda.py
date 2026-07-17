"""CUDA memory resources (design §5.2).

`RmmMemoryResource` wraps any `rmm.mr.DeviceMemoryResource`, translating
streams through the CUDA stream protocol. `CudaRuntimeMemoryResource` is
plain cudaMalloc/cudaFree — cudaMallocAsync/cudaFreeAsync when the driver
supports memory pools — via ctypes on libcudart with no third-party
dependency. Both are thin CUDA bindings of the shared GPU shim
(`devmm._runtimes._gpulib`, design §4.2), unit-testable without hardware
over an injected `GpuApi` (design §9).
"""

from __future__ import annotations

from typing import Any

from devmm._core.stream import Stream
from devmm._runtimes._gpulib import (
    GpuRuntimeMemoryResource,
    RmmLikeMemoryResource,
    rmm_stream_class,
)
from devmm._runtimes.cuda import CUDA_PLATFORM

__all__ = [
    "CudaRuntimeMemoryResource",
    "RmmMemoryResource",
]


def _rmm_stream_class() -> Any:
    """Seam for tests: rmm's Stream class, imported on demand (design §5.2)."""
    return rmm_stream_class(
        "rmm is not importable; install the devmm[cuda] extra to use RmmMemoryResource"
    )


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
