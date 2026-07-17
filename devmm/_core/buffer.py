"""`DeviceBuffer`: an owning, untyped, stream-ordered allocation with weakref-based GC safety net
and deterministic scope-based release (§3.5).
"""

from __future__ import annotations

import weakref
from types import TracebackType

from devmm._core.device import Device, DeviceType
from devmm._core.memory_resource import DeviceMemoryResource
from devmm._core.stream import Stream
from devmm._runtimes import _hostcopy
from devmm._runtimes._discovery import runtime_for
from devmm._runtimes.base import CopyKind


class DeviceBuffer:
    """An owning, untyped, stream-ordered allocation (design §3.5).

    Lifetime rules:

    1. Deallocation is stream-ordered on the allocation stream by default;
       `free(stream=...)` deallocates on a different stream when the caller
       knows the dependency structure (the rmm contract).
    2. GC safety net via `weakref.finalize`, never `__del__`: robust against
       reference cycles and interpreter-shutdown ordering. The finalizer
       captures only `(mr, ptr, nbytes, stream)` — never `self`.
    3. `free()` is idempotent; use-after-free raises through the `closed`
       flag.
    4. `mr` is a strong reference (and adaptors hold their upstream
       strongly), so the full allocator chain outlives every allocation.
    """

    ptr: int
    nbytes: int
    device: Device
    stream: Stream
    mr: DeviceMemoryResource

    def __init__(self, nbytes: int, *, mr: DeviceMemoryResource, stream: Stream) -> None:
        if stream.device != mr.device:
            raise ValueError(
                f"stream device {stream.device} does not match memory resource device {mr.device}"
            )
        ptr = mr.allocate(nbytes, stream)
        # Registered immediately after allocation so an async exception in
        # between cannot leak the pointer. The bound method keeps `mr`
        # alive; none of the captured arguments can reference the buffer, so
        # cycles never keep the allocation alive and the finalizer never
        # resurrects `self` (design §3.5).
        self._finalizer = weakref.finalize(self, mr.deallocate, ptr, nbytes, stream)
        self.ptr = ptr
        self.nbytes = nbytes
        self.device = mr.device
        self.stream = stream
        self.mr = mr

    @property
    def closed(self) -> bool:
        """True once the allocation has been released (or its release began)."""
        return not self._finalizer.alive

    def free(self, stream: Stream | None = None) -> None:
        """Release the allocation; safe to call more than once.

        Deallocation is ordered on the allocation stream unless `stream`
        names another stream on the same device.
        """
        if stream is None:
            # Runs the finalizer callback — deallocate on the allocation
            # stream — exactly once; dead finalizers are a no-op.
            self._finalizer()
            return
        if stream.device != self.device:
            raise ValueError(
                f"cannot deallocate on stream device {stream.device}: "
                f"the buffer lives on {self.device}"
            )
        if self._finalizer.detach() is None:
            return
        self.mr.deallocate(self.ptr, self.nbytes, stream)

    def __enter__(self) -> DeviceBuffer:
        self._check_open("__enter__")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.free()

    def copy_from_host(self, data: bytes | memoryview, stream: Stream | None = None) -> None:
        """Copy host bytes into the start of the buffer.

        A byte-level testing/bootstrap helper, not a transfer API (design
        §3.5). `stream` is accepted for the stream-ordered calling
        convention; the host-memory path is synchronous.
        """
        self._check_open("copy_from_host")
        self._check_host_resident("copy_from_host")
        view = memoryview(data)
        # C-contiguity specifically: the staging copy serialises in C order,
        # so accepting a merely Fortran-contiguous view would silently
        # reorder the bytes it copies.
        if not view.c_contiguous:
            raise ValueError("copy_from_host requires a C-contiguous source buffer")
        if view.nbytes > self.nbytes:
            raise ValueError(f"cannot copy {view.nbytes} bytes into a {self.nbytes}-byte buffer")
        if view.nbytes:
            # HOST_TO_HOST because `_check_host_resident` above confines the
            # helpers to host-resident buffers.
            _hostcopy.copy_from_host(
                runtime_for(self.device),
                self.ptr,
                view,
                CopyKind.HOST_TO_HOST,
                self.stream if stream is None else stream,
            )

    def copy_to_host(self, stream: Stream | None = None) -> bytes:
        """Read the buffer's full contents back as bytes (see `copy_from_host`)."""
        self._check_open("copy_to_host")
        self._check_host_resident("copy_to_host")
        if self.nbytes == 0:
            return b""
        return _hostcopy.copy_to_host(
            runtime_for(self.device),
            self.ptr,
            self.nbytes,
            CopyKind.HOST_TO_HOST,
            self.stream if stream is None else stream,
        )

    def _check_open(self, operation: str) -> None:
        if self.closed:
            raise ValueError(f"{operation} on a freed DeviceBuffer (use-after-free)")

    def _check_host_resident(self, operation: str) -> None:
        # The helpers stage through host memory and issue a HOST_TO_HOST
        # memcpy; device-resident buffers need the runtime's device-transfer
        # copy kinds, which no wired runtime implements (design §4.1).
        if self.device.type is not DeviceType.CPU:
            raise NotImplementedError(
                f"{operation} supports only host-resident (cpu) buffers, not {self.device}"
            )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(ptr={self.ptr:#x}, nbytes={self.nbytes}, "
            f"device={self.device}, closed={self.closed})"
        )
