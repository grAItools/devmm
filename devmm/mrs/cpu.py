"""CPU memory resources (design §5.1).

`BytearrayMemoryResource` is the paranoid pure-Python reference MR: backing
storage is a `bytearray` pinned in place via a ctypes buffer export.
`MallocMemoryResource` is the default CPU MR: exact-alignment allocation from
the platform C runtime (`posix_memalign`/`free` on POSIX,
`_aligned_malloc`/`_aligned_free` on Windows), tracking the allocating family
per pointer because the Windows pair is not `free`-compatible.
"""

from __future__ import annotations

import ctypes
import functools
import sys
import threading
from typing import Protocol

from devmm._core.device import Device, DeviceType
from devmm._core.memory_resource import DeviceMemoryResource
from devmm._core.stream import Stream

__all__ = [
    "BytearrayMemoryResource",
    "MallocMemoryResource",
]

_CPU_DEVICE = Device(DeviceType.CPU)


def _check_cpu_device(device: Device, mr_name: str) -> None:
    if device.type is not DeviceType.CPU:
        raise ValueError(f"{mr_name} requires a cpu device, got {device}")


def _check_alignment(alignment: int) -> None:
    if alignment < 1 or alignment & (alignment - 1):
        raise ValueError(f"alignment must be a power of two >= 1, got {alignment}")


class _PinnedBytearray(bytearray):
    """Backing store for `BytearrayMemoryResource`.

    Plain `bytearray` does not support weak references; the release contract
    (the backing store dies as soon as `deallocate` drops it) is observable
    only through a weakref, so the store carries a weakref slot.
    """

    __slots__ = ("__weakref__",)


class BytearrayMemoryResource(DeviceMemoryResource):
    """Pure-Python CPU MR backed by pinned bytearrays (design §5.1).

    `allocate` over-allocates by `alignment - 1` bytes, pins the bytearray
    against resize/relocation via a ctypes buffer export, and returns the
    first `alignment`-aligned address inside it. The paranoid default that
    works on any CPython anywhere, and the reference MR for the conformance
    suite.

    `guaranteed_alignment()` stays at the ABC's conservative 1: alignment is
    achieved by offsetting into unaligned storage, never promised by the
    underlying allocator (design §5.1).
    """

    def __init__(self, device: Device = _CPU_DEVICE, *, alignment: int = 1) -> None:
        _check_cpu_device(device, type(self).__name__)
        _check_alignment(alignment)
        self.device = device
        self._alignment = alignment
        self._lock = threading.Lock()
        # ptr -> (backing store, ctypes pin). The pin's buffer export blocks
        # resizing/relocation; the dict keeps both alive until deallocate.
        self._live: dict[int, tuple[_PinnedBytearray, ctypes.Array[ctypes.c_char]]] = {}

    def allocate(self, nbytes: int, stream: Stream) -> int:
        if nbytes < 0:
            raise ValueError(f"cannot allocate a negative size ({nbytes} bytes)")
        # max(nbytes, 1): a zero-byte request still gets a unique, freeable
        # pointer (distinct live bytearrays never share an address).
        try:
            backing = _PinnedBytearray(max(nbytes, 1) + self._alignment - 1)
        except MemoryError as exc:
            raise MemoryError(
                f"failed to allocate {nbytes} bytes on {self.device} in {self!r}"
            ) from exc
        pin = (ctypes.c_char * len(backing)).from_buffer(backing)
        base = ctypes.addressof(pin)
        ptr = base + ((-base) % self._alignment)
        with self._lock:
            self._live[ptr] = (backing, pin)
        return ptr

    def deallocate(self, ptr: int, nbytes: int, stream: Stream) -> None:
        with self._lock:
            entry = self._live.pop(ptr, None)
        if entry is None:
            raise ValueError(
                f"pointer {ptr:#x} is not a live allocation of {self!r} "
                "(double-free, or a pointer this MR never returned)"
            )

    def _debug_live_count(self) -> int:
        """Testing hook: allocations currently tracked (0 == tables empty)."""
        with self._lock:
            return len(self._live)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(device={self.device}, alignment={self._alignment})"


class _AllocationFamily(Protocol):
    """One aligned-alloc/free pairing from a platform C runtime.

    `MallocMemoryResource` records the family per pointer and frees through
    the recorded one: a Windows `_aligned_malloc` pointer is only valid for
    `_aligned_free`, never `free` (design §5.1).
    """

    @property
    def name(self) -> str: ...

    def alloc(self, nbytes: int, alignment: int) -> int:
        """Return an `alignment`-aligned pointer to `nbytes` bytes; 0 on failure."""
        ...

    def free(self, ptr: int) -> None: ...


class _PosixFamily:
    """`posix_memalign`/`free` from the process's libc."""

    name = "posix"

    def __init__(self) -> None:
        libc = ctypes.CDLL(None, use_errno=True)
        self._posix_memalign = libc.posix_memalign
        self._posix_memalign.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_size_t,
            ctypes.c_size_t,
        ]
        self._posix_memalign.restype = ctypes.c_int
        self._free = libc.free
        self._free.argtypes = [ctypes.c_void_p]
        self._free.restype = None

    def alloc(self, nbytes: int, alignment: int) -> int:
        out = ctypes.c_void_p()
        # posix_memalign requires alignment to be a power-of-two multiple of
        # sizeof(void*); rounding up only strengthens the caller's request.
        rc = self._posix_memalign(
            ctypes.byref(out), max(alignment, ctypes.sizeof(ctypes.c_void_p)), nbytes
        )
        if rc != 0 or not out.value:
            return 0
        return out.value

    def free(self, ptr: int) -> None:
        self._free(ctypes.c_void_p(ptr))


class _WindowsFamily:
    """`_aligned_malloc`/`_aligned_free` from the Microsoft C runtime."""

    name = "windows"

    def __init__(self) -> None:
        crt = ctypes.CDLL("msvcrt", use_errno=True)
        self._aligned_malloc = crt._aligned_malloc
        self._aligned_malloc.argtypes = [ctypes.c_size_t, ctypes.c_size_t]
        self._aligned_malloc.restype = ctypes.c_void_p
        self._aligned_free = crt._aligned_free
        self._aligned_free.argtypes = [ctypes.c_void_p]
        self._aligned_free.restype = None

    def alloc(self, nbytes: int, alignment: int) -> int:
        ptr = self._aligned_malloc(nbytes, alignment)
        return 0 if ptr is None else int(ptr)

    def free(self, ptr: int) -> None:
        self._aligned_free(ctypes.c_void_p(ptr))


@functools.cache
def _native_family() -> _AllocationFamily:
    """The platform's aligned-allocation family, loaded once per process."""
    if sys.platform == "win32":
        return _WindowsFamily()
    return _PosixFamily()


class MallocMemoryResource(DeviceMemoryResource):
    """Default CPU MR: exact-alignment allocation from the platform C runtime
    (design §5.1).

    Every pointer is `alignment`-aligned by the allocator itself, so
    `guaranteed_alignment()` reports the configured value exactly and
    `empty()`-style callers never over-allocate. The default of 64 is the
    cache-line base alignment the CPU `DeviceOptimal` layout policy requests
    (`devmm._core.layout`).
    """

    def __init__(self, device: Device = _CPU_DEVICE, *, alignment: int = 64) -> None:
        _check_cpu_device(device, type(self).__name__)
        _check_alignment(alignment)
        self.device = device
        self._alignment = alignment
        self._lock = threading.Lock()
        # ptr -> (nbytes, allocating family): deallocate validates the size
        # and dispatches to the matching native free.
        self._live: dict[int, tuple[int, _AllocationFamily]] = {}

    def allocate(self, nbytes: int, stream: Stream) -> int:
        if nbytes < 0:
            raise ValueError(f"cannot allocate a negative size ({nbytes} bytes)")
        family = _native_family()
        # max(nbytes, 1): C permits size-0 allocation to return NULL or a
        # reusable address, either of which would collide in the tracking
        # table; one byte buys a unique, freeable pointer.
        ptr = family.alloc(max(nbytes, 1), self._alignment)
        if ptr == 0:
            raise MemoryError(f"failed to allocate {nbytes} bytes on {self.device} in {self!r}")
        with self._lock:
            self._live[ptr] = (nbytes, family)
        return ptr

    def deallocate(self, ptr: int, nbytes: int, stream: Stream) -> None:
        with self._lock:
            entry = self._live.get(ptr)
            if entry is None:
                raise ValueError(
                    f"pointer {ptr:#x} is not a live allocation of {self!r} "
                    "(double-free, or a pointer this MR never returned)"
                )
            recorded, family = entry
            if nbytes != recorded:
                # The allocation stays live: a size-mismatched free is caller
                # confusion, and releasing anyway would turn it into a
                # use-after-free elsewhere.
                raise ValueError(
                    f"size mismatch freeing pointer {ptr:#x} in {self!r}: "
                    f"allocated {recorded} bytes, deallocate got {nbytes}"
                )
            del self._live[ptr]
        family.free(ptr)

    def guaranteed_alignment(self) -> int:
        return self._alignment

    def _debug_live_count(self) -> int:
        """Testing hook: allocations currently tracked (0 == tables empty)."""
        with self._lock:
            return len(self._live)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(device={self.device}, alignment={self._alignment})"
