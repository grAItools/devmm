"""ROCm memory resources (design §5.3).

`HipmmMemoryResource` wraps hipMM's Python resources — hipMM installs under
the module name `rmm`, so the §4.2 platform disambiguation applies before
anything trusts that import. `HipRuntimeMemoryResource` is plain
hipMalloc/hipFree — hipMallocAsync/hipFreeAsync when the driver supports
memory pools — via ctypes on libamdhip64, the fallback while hipMM's Python
port completes upstream. Both are thin HIP bindings of the shared GPU shim
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
from devmm._runtimes.rocm import HIP_PLATFORM

__all__ = [
    "HipRuntimeMemoryResource",
    "HipmmMemoryResource",
]


def _rmm_stream_class() -> Any:
    """Seam for tests: hipMM's Stream class, imported on demand (design §5.3)."""
    return rmm_stream_class(
        "hipMM's rmm module is not importable; install the devmm[rocm] extra "
        "to use HipmmMemoryResource"
    )


class HipRuntimeMemoryResource(GpuRuntimeMemoryResource):
    """The "just hipMalloc" MR (design §5.3), the mirror image of
    `CudaRuntimeMemoryResource`; semantics in `GpuRuntimeMemoryResource`."""

    platform = HIP_PLATFORM


class HipmmMemoryResource(RmmLikeMemoryResource):
    """The wrapper over hipMM's `rmm`-named Python resources (design §5.3),
    same shape as `RmmMemoryResource`; semantics in `RmmLikeMemoryResource`."""

    platform = HIP_PLATFORM

    def _translated_stream(self, stream: Stream) -> Any:
        return _rmm_stream_class()(stream)
