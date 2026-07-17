"""NEP-49 handler installation (design §6): `install(mr)` builds a
`PyDataMem_Handler` from ctypes thunks over `mr` and `PyDataMem_SetHandler`s
it, so every NumPy array data allocation in the process routes through the
devmm MR chain — `install(StatisticsAdaptor(MallocMemoryResource()))` yields
allocation statistics over every NumPy array.

Lifetime: NumPy keeps a per-array reference to the allocating handler
capsule, so arrays allocated while installed stay freeable (through `mr`)
long after uninstall. The state behind a capsule — the ctypes struct, the
MR, the ptr->size table — is therefore retired by the capsule's destructor,
not by `uninstall()`.
"""

from __future__ import annotations

import ctypes
import logging
import threading
from typing import Any

from devmm import _nep49
from devmm._core.device import DeviceType
from devmm._core.memory_resource import DeviceMemoryResource
from devmm._core.stream import CpuStream, Stream
from devmm.integrations._support import Installer, ensure_no_cycle
from devmm.mrs.cpu import NumpyHandlerMemoryResource

__all__ = ["install"]

_LOGGER = logging.getLogger("devmm.mr")

_HANDLER_NAME = b"devmm"


class _HandlerState:
    """Everything one installed handler keeps alive: the struct the capsule
    points at, the target MR, and the ptr->size table realloc/free need
    (the recorded size is authoritative — it is what `allocate` passed to
    the MR; NumPy recomputes the size it passes to free)."""

    __slots__ = ("handler", "lock", "mr", "sizes", "stream")

    def __init__(self, mr: DeviceMemoryResource) -> None:
        self.mr = mr
        self.stream: Stream = CpuStream(mr.device)
        self.lock = threading.Lock()
        self.sizes: dict[int, int] = {}
        handler = _nep49.PyDataMemHandler()
        handler.name = _HANDLER_NAME
        handler.version = _nep49.HANDLER_ABI_VERSION
        allocator = handler.allocator
        # ctx doubles as the state key: every callback receives it and looks
        # its state up in _LIVE, so four module-global thunks serve every
        # install ever made.
        allocator.ctx = ctypes.addressof(handler)
        allocator.malloc = _MALLOC
        allocator.calloc = _CALLOC
        allocator.realloc = _REALLOC
        allocator.free = _FREE
        self.handler = handler

    @property
    def address(self) -> int:
        return ctypes.addressof(self.handler)


# Live states keyed by handler-struct address (== the capsule data pointer
# == the ctx the callbacks receive). Entries are added at install and
# removed by the capsule destructor once no array or installer can reach
# the handler any more.
_LIVE: dict[int, _HandlerState] = {}


def _allocate(state: _HandlerState, nbytes: int) -> int:
    try:
        ptr = state.mr.allocate(nbytes, state.stream)
    except MemoryError:
        # NULL is the C-side spelling of allocation failure; NumPy raises
        # its own MemoryError from it.
        return 0
    except Exception:
        # A C caller cannot see a Python exception; NULL at least fails the
        # allocation cleanly instead of crashing the process.
        _LOGGER.exception("devmm NEP-49 handler: allocating %d bytes failed", nbytes)
        return 0
    with state.lock:
        state.sizes[ptr] = nbytes
    return ptr


def _release(state: _HandlerState, ptr: int, nbytes: int) -> None:
    try:
        state.mr.deallocate(ptr, nbytes, state.stream)
    except Exception:
        _LOGGER.exception("devmm NEP-49 handler: freeing pointer %#x failed", ptr)


def _malloc_impl(ctx: int | None, size: int) -> int:
    state = _LIVE.get(ctx or 0)
    if state is None:
        return 0
    return _allocate(state, size)


def _calloc_impl(ctx: int | None, nelem: int, elsize: int) -> int:
    state = _LIVE.get(ctx or 0)
    if state is None:
        return 0
    nbytes = nelem * elsize
    ptr = _allocate(state, nbytes)
    if ptr:
        ctypes.memset(ptr, 0, nbytes)
    return ptr


def _realloc_impl(ctx: int | None, ptr: int | None, new_size: int) -> int:
    state = _LIVE.get(ctx or 0)
    if state is None:
        return 0
    if not ptr:
        # C realloc semantics: a NULL pointer means plain allocation.
        return _allocate(state, new_size)
    with state.lock:
        old_size = state.sizes.get(ptr)
    if old_size is None:
        _LOGGER.error("devmm NEP-49 handler: realloc of unknown pointer %#x", ptr)
        return 0
    new_ptr = _allocate(state, new_size)
    if not new_ptr:
        # C realloc semantics: on failure the old allocation stays valid.
        return 0
    ctypes.memmove(new_ptr, ptr, min(old_size, new_size))
    with state.lock:
        state.sizes.pop(ptr, None)
    _release(state, ptr, old_size)
    return new_ptr


def _free_impl(ctx: int | None, ptr: int | None, size: int) -> None:
    state = _LIVE.get(ctx or 0)
    if state is None or not ptr:
        return
    with state.lock:
        recorded = state.sizes.pop(ptr, None)
    if recorded is None:
        _LOGGER.error("devmm NEP-49 handler: free of a pointer it never allocated: %#x", ptr)
        return
    _release(state, ptr, recorded)


# Module-scope references are load-bearing: NumPy holds raw function
# pointers to these thunks inside handler structs for as long as any array
# allocated under them lives; a garbage-collected thunk is a segfault.
_MALLOC = _nep49.MallocFunc(_malloc_impl)
_CALLOC = _nep49.CallocFunc(_calloc_impl)
_REALLOC = _nep49.ReallocFunc(_realloc_impl)
_FREE = _nep49.FreeFunc(_free_impl)


def _destroy_handler_capsule(
    capsule_ptr: int | None,
    _is_initialized: Any = _nep49.py_is_initialized,
    _is_valid: Any = _nep49.capsule_is_valid_raw,
    _get_pointer: Any = _nep49.capsule_pointer_raw,
    _live: dict[int, _HandlerState] = _LIVE,
    _name: bytes = _nep49.HANDLER_CAPSULE_NAME,
) -> None:
    """Capsule destructor: the last reference to a devmm handler capsule
    died — every array allocated under it has been freed and the installer
    dropped — so retire the state that kept the struct and MR alive.

    Everything the body needs is bound as a default so the destructor stays
    callable even after this module's globals are torn down at interpreter
    shutdown.
    """
    if not capsule_ptr or not _is_initialized() or not _is_valid(capsule_ptr, _name):
        return
    address = _get_pointer(capsule_ptr, _name)
    if address:
        _live.pop(address, None)


_HANDLER_CAPSULE_DESTRUCTOR = _nep49.PyCapsuleDestructor(_destroy_handler_capsule)

# Immortalize the thunks: an array (hence a handler capsule, hence these
# function pointers) may outlive this module's teardown at interpreter
# shutdown.
_nep49.py_inc_ref(_MALLOC)
_nep49.py_inc_ref(_CALLOC)
_nep49.py_inc_ref(_REALLOC)
_nep49.py_inc_ref(_FREE)
_nep49.py_inc_ref(_HANDLER_CAPSULE_DESTRUCTOR)


def _new_handler_capsule(state: _HandlerState) -> Any:
    _LIVE[state.address] = state
    try:
        return _nep49.capsule_new(
            state.address, _nep49.HANDLER_CAPSULE_NAME, _HANDLER_CAPSULE_DESTRUCTOR
        )
    except BaseException:  # pragma: no cover — PyCapsule_New fails only on memory exhaustion
        _LIVE.pop(state.address, None)
        raise


def install(mr: DeviceMemoryResource) -> Installer:
    """Install a NEP-49 data-memory handler over `mr` (design §6).

    An explicit call, never an import side effect. The returned `Installer`
    restores the previously installed handler;
    `numpy.core.multiarray.get_handler_name()` (or the `numpy._core` 2.x
    spelling) reports ``"devmm"`` while installed. Installing an MR that
    itself consumes NumPy's handler (`NumpyHandlerMemoryResource`) is
    refused as a direct cycle. NumPy support is pinned to
    `devmm._nep49.SUPPORTED_NUMPY_RANGE`.
    """
    ensure_no_cycle(mr, NumpyHandlerMemoryResource, "NumPy")
    if mr.device.type is not DeviceType.CPU:
        raise ValueError(
            f"a NumPy data-memory handler allocates host memory; the "
            f"installed MR must live on the cpu device, got {mr.device}"
        )
    api = _nep49.load_api()
    capsule = _new_handler_capsule(_HandlerState(mr))
    previous = api.set_handler(capsule)

    def _restore() -> None:
        api.set_handler(previous)

    return Installer("NumPy", _restore)
