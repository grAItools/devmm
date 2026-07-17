"""ROCm runtime over libamdhip64 via ctypes; hipMM-or-hipMalloc default MR (§4).

All control flow lives in the shared GPU shim (`devmm._runtimes._gpulib`,
design §4.2); this module binds the HIP symbol table, error type and magic
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


class HipError(GpuError):
    """A libamdhip64 call returned a nonzero `hipError_t` (see `GpuError`)."""

    error_label = "HIP"


if sys.platform == "win32":
    _LIBAMDHIP_NAMES = ("amdhip64_7.dll", "amdhip64_6.dll", "amdhip64.dll")
else:
    # Sonames follow the ROCm major version, newest-first, then the
    # unversioned dev-symlink spelling.
    _LIBAMDHIP_NAMES = (
        "libamdhip64.so.7",
        "libamdhip64.so.6",
        "libamdhip64.so.5",
        "libamdhip64.so",
    )


HIP_PLATFORM = GpuPlatform(
    name="rocm",
    device_type=DeviceType.ROCM,
    error=HipError,
    symbols=GpuSymbols(
        get_error_string="hipGetErrorString",
        get_device_count="hipGetDeviceCount",
        get_device="hipGetDevice",
        set_device="hipSetDevice",
        get_device_attribute="hipDeviceGetAttribute",
        malloc="hipMalloc",
        free="hipFree",
        malloc_async="hipMallocAsync",
        free_async="hipFreeAsync",
        stream_create="hipStreamCreate",
        stream_destroy="hipStreamDestroy",
        stream_synchronize="hipStreamSynchronize",
        memcpy_async="hipMemcpyAsync",
        event_create_with_flags="hipEventCreateWithFlags",
        event_record="hipEventRecord",
        stream_wait_event="hipStreamWaitEvent",
        event_destroy="hipEventDestroy",
    ),
    # hipErrorNotSupported: what the async entry points report when the
    # loaded libamdhip64 predates them (ROCm < 5.2).
    not_supported_status=801,
    # hipEventDisableTiming.
    event_disable_timing=0x2,
    # hipDeviceAttributeMemoryPoolsSupported (hip_runtime_api.h, the
    # renumbered CUDA-compatible section of `hipDeviceAttribute_t`): the
    # driver capability behind hipMallocAsync/hipFreeAsync.
    memory_pools_attribute=88,
    # HIP's magic default-stream handles (design §3.2): the null stream
    # doubles as the legacy default — the DLPack consumer-stream table in
    # `_dlpack/export.py` maps ROCm's legacy default to 0 for the same
    # reason — and hipStreamPerThread is (hipStream_t)2.
    sentinel_handles={DEFAULT: 0x0, LEGACY_DEFAULT: 0x0, PER_THREAD_DEFAULT: 0x2},
    library_alias="amdhip64",
    library_names=_LIBAMDHIP_NAMES,
    library_hint=(
        "libamdhip64 is not loadable; install the ROCm HIP runtime to use the rocm runtime"
    ),
)


class HipStream(GpuStream):
    """A HIP stream handle bound to a `GpuApi` (design §3.2)."""


def _import_rmm() -> ModuleType | None:
    """Seam for tests: the ROCm-verified `rmm` module, else None (§4.2)."""
    return import_rmm_for("rocm")


class HipRuntime(GpuRuntime):
    """The AMD platform's `DeviceRuntime` (design §4.1) over an injected
    `GpuApi` — the process's libamdhip64 by default.

    Default-MR chain (design §4.1): `HipmmMemoryResource` over hipMM's
    per-device resource when a ROCm-targeting rmm-named module imports
    (§4.2), else `HipRuntimeMemoryResource` (async variant when the driver
    supports memory pools).
    """

    name = "rocm"
    device_types = frozenset({DeviceType.ROCM})
    platform = HIP_PLATFORM
    _stream_type = HipStream

    def default_memory_resource(self, device: Device) -> DeviceMemoryResource:
        self._check_device(device)
        # Imported lazily: devmm.mrs.rocm imports this module for the
        # platform table, so a module-scope import would be circular.
        from devmm.mrs.rocm import HipmmMemoryResource, HipRuntimeMemoryResource

        rmm = _import_rmm()
        if rmm is not None:
            # Users who already configured hipMM (pools, reinitialize, ...)
            # get that configuration for free (design §5.3).
            return HipmmMemoryResource(rmm.mr.get_per_device_resource(device.index), device)
        return HipRuntimeMemoryResource(device, async_alloc="auto", api=self.api)
