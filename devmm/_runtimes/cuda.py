"""CUDA runtime over libcudart via ctypes; rmm-or-cudaMalloc default MR (§4).

All control flow lives in the shared GPU shim (`devmm._runtimes._gpulib`,
design §4.2); this module binds the CUDA symbol table, error type and magic
handles, plus the platform's default-MR chain.
"""

from __future__ import annotations

import sys
from types import ModuleType

from devmm._core.device import Device, DeviceType
from devmm._core.memory_resource import DeviceMemoryResource
from devmm._core.stream import DEFAULT, LEGACY_DEFAULT, PER_THREAD_DEFAULT
from devmm._runtimes._gpulib import (
    GpuError,
    GpuPlatform,
    GpuRuntime,
    GpuStream,
    GpuSymbols,
    import_rmm_for,
)


class CudaError(GpuError):
    """A libcudart call returned a nonzero `cudaError_t` (see `GpuError`)."""

    error_label = "CUDA"


if sys.platform == "win32":
    _LIBCUDART_NAMES = ("cudart64_13.dll", "cudart64_12.dll", "cudart64_110.dll")
else:
    # Versioned names newest-first, then the unversioned dev-symlink
    # spelling (present only where the toolkit's dev files are installed).
    _LIBCUDART_NAMES = ("libcudart.so.13", "libcudart.so.12", "libcudart.so.11", "libcudart.so")


CUDA_PLATFORM = GpuPlatform(
    name="cuda",
    device_type=DeviceType.CUDA,
    error=CudaError,
    symbols=GpuSymbols(
        get_error_string="cudaGetErrorString",
        get_device_count="cudaGetDeviceCount",
        get_device="cudaGetDevice",
        set_device="cudaSetDevice",
        get_device_attribute="cudaDeviceGetAttribute",
        malloc="cudaMalloc",
        free="cudaFree",
        malloc_async="cudaMallocAsync",
        free_async="cudaFreeAsync",
        stream_create="cudaStreamCreate",
        stream_destroy="cudaStreamDestroy",
        stream_synchronize="cudaStreamSynchronize",
        memcpy_async="cudaMemcpyAsync",
        event_create_with_flags="cudaEventCreateWithFlags",
        event_record="cudaEventRecord",
        stream_wait_event="cudaStreamWaitEvent",
        event_destroy="cudaEventDestroy",
    ),
    # cudaErrorNotSupported: what the async entry points report when the
    # loaded libcudart predates them (CUDA < 11.2).
    not_supported_status=801,
    # cudaEventDisableTiming.
    event_disable_timing=0x2,
    # cudaDevAttrMemoryPoolsSupported: the driver capability behind
    # cudaMallocAsync/cudaFreeAsync.
    memory_pools_attribute=115,
    # The platform's magic default-stream handles, mirroring rmm's
    # sentinels (design §3.2): cudaStreamDefault / cudaStreamLegacy /
    # cudaStreamPerThread.
    sentinel_handles={DEFAULT: 0x0, LEGACY_DEFAULT: 0x1, PER_THREAD_DEFAULT: 0x2},
    library_alias="cudart",
    library_names=_LIBCUDART_NAMES,
    library_hint=(
        "libcudart is not loadable; install the CUDA toolkit runtime "
        "(or a wheel bundling it) to use the cuda runtime"
    ),
)


class CudaStream(GpuStream):
    """A CUDA stream handle bound to a `GpuApi` (design §3.2)."""


def _import_rmm() -> ModuleType | None:
    """Seam for tests: the CUDA-verified `rmm` module, else None (§4.2)."""
    return import_rmm_for("cuda")


class CudaRuntime(GpuRuntime):
    """The NVIDIA platform's `DeviceRuntime` (design §4.1) over an injected
    `GpuApi` — the process's libcudart by default.

    Default-MR chain (design §4.1): `RmmMemoryResource` over rmm's
    per-device resource when a CUDA-targeting rmm imports (§4.2), else
    `CudaRuntimeMemoryResource` (async variant when the driver supports
    memory pools).
    """

    name = "cuda"
    device_types = frozenset({DeviceType.CUDA})
    platform = CUDA_PLATFORM
    _stream_type = CudaStream

    def default_memory_resource(self, device: Device) -> DeviceMemoryResource:
        self._check_device(device)
        # Imported lazily: devmm.mrs.cuda imports this module for the
        # platform table, so a module-scope import would be circular.
        from devmm.mrs.cuda import CudaRuntimeMemoryResource, RmmMemoryResource

        rmm = _import_rmm()
        if rmm is not None:
            # Users who already configured rmm (pools, reinitialize, ...)
            # get that configuration for free (design §5.2).
            return RmmMemoryResource(rmm.mr.get_per_device_resource(device.index), device)
        return CudaRuntimeMemoryResource(device, async_alloc="auto", api=self.api)
