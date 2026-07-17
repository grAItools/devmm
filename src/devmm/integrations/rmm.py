"""Registry bridge into the rmm module (design §6): `install(mr)` replaces
rmm's per-device resource for `mr.device` with an rmm
`CallbackMemoryResource` forwarding to `mr`.

devmm's own registry (design §3.4) deliberately never touches rmm's global
state; this explicit bridge is how the two registries get linked. hipMM
installs under the same module name, so the §4.2 platform check runs before
anything trusts the import.
"""

from __future__ import annotations

from typing import Any

from devmm._core.device import DeviceType
from devmm._core.memory_resource import DeviceMemoryResource
from devmm._runtimes._gpulib import RmmLikeMemoryResource, import_rmm_for
from devmm._runtimes.base import RuntimeUnavailableError
from devmm.integrations._support import (
    ForeignHandleStream,
    Installer,
    ensure_no_cycle,
    stream_handle_of,
)

__all__ = ["install"]

_PLATFORM_NAMES = {DeviceType.CUDA: "cuda", DeviceType.ROCM: "rocm"}


def _rmm_module_for(platform_name: str) -> Any:
    module = import_rmm_for(platform_name)
    if module is None:
        raise RuntimeUnavailableError(
            f"no rmm module targeting {platform_name} is importable; install "
            f"the devmm[{platform_name}] extra to bridge devmm into rmm"
        )
    return module


def install(mr: DeviceMemoryResource) -> Installer:
    """Make rmm's per-device resource for `mr.device` allocate through `mr`
    via `rmm.mr.CallbackMemoryResource` (design §6).

    The returned `Installer` restores the previous per-device resource.
    Installing an MR that itself consumes an rmm resource
    (`RmmMemoryResource`/`HipmmMemoryResource`) is refused — pointed back at
    rmm's registry, the callback would allocate from itself.
    """
    ensure_no_cycle(mr, RmmLikeMemoryResource, "rmm")
    platform_name = _PLATFORM_NAMES.get(mr.device.type)
    if platform_name is None:
        raise ValueError(f"rmm serves cuda/rocm devices; the installed MR lives on {mr.device}")
    module = _rmm_module_for(platform_name)

    # rmm's callback trampoline passes (nbytes, stream) / (ptr, nbytes,
    # stream); the defaults keep older callback conventions (no stream
    # argument) on the platform default stream.
    def _allocate(nbytes: int, stream: Any = None) -> int:
        return mr.allocate(nbytes, ForeignHandleStream(mr.device, stream_handle_of(stream)))

    def _deallocate(ptr: int, nbytes: int, stream: Any = None) -> None:
        mr.deallocate(ptr, nbytes, ForeignHandleStream(mr.device, stream_handle_of(stream)))

    index = mr.device.index
    previous = module.mr.get_per_device_resource(index)
    module.mr.set_per_device_resource(
        index, module.mr.CallbackMemoryResource(_allocate, _deallocate)
    )

    def _restore() -> None:
        module.mr.set_per_device_resource(index, previous)

    return Installer("rmm", _restore)
