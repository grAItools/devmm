"""`DevmmEMMPlugin`: make Numba allocate through the current devmm MR
(design §5.4, §6).

The plugin subclasses Numba's `HostOnlyCUDAMemoryManager`, so the class is
built lazily on first attribute access — importing this module never imports
numba. Numba's ``NUMBA_CUDA_MEMORY_MANAGER=devmm.integrations.numba`` env
hook reads the module-level ``_numba_memory_manager`` name, served by the
same lazy hook.

The consumer direction — allocating *from* a standalone EMM plugin — is
deliberately not shipped (design §5.4): plugins are written to be plugged
into Numba. The rare user who needs it can wrap a plugin instance in a
`CallbackMemoryResource` whose alloc callback calls ``plugin.memalloc(nbytes)``
and stashes the returned `MemoryPointer` in a ptr-keyed dict, and whose
dealloc callback drops the entry (Numba's finalizer machinery frees on
refcount zero).
"""

from __future__ import annotations

import ctypes
import importlib
from collections.abc import Callable
from typing import Any

from devmm._core.buffer import DeviceBuffer
from devmm._core.device import Device, DeviceType
from devmm._core.registry import get_current_memory_resource
from devmm._runtimes.base import RuntimeUnavailableError
from devmm.integrations._support import ForeignHandleStream, Installer

# DevmmEMMPlugin is served lazily by the module __getattr__ below, so the
# static F822 "undefined name" check cannot see it.
__all__ = ["DevmmEMMPlugin", "install"]  # noqa: F822

# The Numba-facing spelling next to the public class name: both resolve to
# the lazily built plugin class.
_LAZY_EXPORTS = ("DevmmEMMPlugin", "_numba_memory_manager")

_plugin_class_cache: type | None = None


def _numba_cuda_module() -> Any:
    """Seam for tests: `numba.cuda`, imported on demand."""
    try:
        return importlib.import_module("numba.cuda")
    except ImportError as exc:
        raise RuntimeUnavailableError(
            "numba is not importable; install the devmm[numba] extra to use the devmm EMM plugin"
        ) from exc


def _make_finalizer(allocations: Any, ptr: int) -> Callable[[], None]:
    """`MemoryPointer` finalizer: frees the buffer when Numba's last
    reference dies.

    Bound to the allocations mapping, not the manager: at teardown Numba may
    reset the context (clearing the mapping) while device arrays still hold
    pointers — then there is nothing left to free here and the buffer's own
    safety net owns the release.
    """

    def finalize() -> None:
        buffer = allocations.pop(ptr, None)
        if buffer is not None:
            buffer.free()

    return finalize


def _build_plugin_class(numba_cuda: Any) -> type:
    # The bases come off the lazily imported numba module, which mypy can
    # only see as Any; the EMM protocol contract is pinned by tests instead.
    class DevmmEMMPlugin(
        numba_cuda.GetIpcHandleMixin,  # type: ignore[misc]
        numba_cuda.HostOnlyCUDAMemoryManager,  # type: ignore[misc]
    ):
        """Numba EMM plugin allocating device memory through the current
        devmm MR (design §5.4): every `memalloc` resolves the registry for
        the context's device, so Numba kernels get pooled/tracked memory and
        devmm statistics cover Numba allocations. Host allocations
        (pinned/mapped) stay with Numba's own implementation.
        """

        def initialize(self) -> None:
            # No native setup: the registry resolves the MR lazily per
            # allocation.
            return None

        def memalloc(self, size: int) -> Any:
            device = Device(DeviceType.CUDA, self.context.device.id)
            mr = get_current_memory_resource(device)
            # The EMM protocol carries no stream, so allocations ride the
            # default stream (handle 0) — the same choice rmm's plugin
            # makes.
            buffer = DeviceBuffer(size, mr=mr, stream=ForeignHandleStream(device, 0))
            self.allocations[buffer.ptr] = buffer
            return numba_cuda.MemoryPointer(
                self.context,
                ctypes.c_uint64(buffer.ptr),
                size,
                finalizer=_make_finalizer(self.allocations, buffer.ptr),
            )

        def get_memory_info(self) -> Any:
            device = Device(DeviceType.CUDA, self.context.device.id)
            info = get_current_memory_resource(device).available_memory()
            if info is None:
                raise NotImplementedError(
                    "the current devmm memory resource does not report available memory"
                )
            free, total = info
            return numba_cuda.MemoryInfo(free=free, total=total)

        @property
        def interface_version(self) -> int:
            return 1

    return DevmmEMMPlugin


def _plugin_class() -> type:
    global _plugin_class_cache
    if _plugin_class_cache is None:
        _plugin_class_cache = _build_plugin_class(_numba_cuda_module())
    return _plugin_class_cache


def __getattr__(name: str) -> Any:
    # PEP 562: build the Numba subclass only when someone asks for it.
    if name in _LAZY_EXPORTS:
        return _plugin_class()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def install() -> Installer:
    """`numba.cuda.set_memory_manager(DevmmEMMPlugin)` (design §6).

    Numba resolves its memory manager at context creation, so install before
    any CUDA context exists; later installs only affect later contexts
    (Numba's own documented constraint). Uninstall restores the previous
    plugin by resetting Numba's module global — `set_memory_manager` cannot
    express "back to the built-in manager" (it instantiates its argument, so
    None is not acceptable input).
    """
    numba_cuda = _numba_cuda_module()
    driver = numba_cuda.cudadrv.driver
    previous = driver._memory_manager
    numba_cuda.set_memory_manager(_plugin_class())

    def _restore() -> None:
        driver._memory_manager = previous

    return Installer("Numba", _restore)
