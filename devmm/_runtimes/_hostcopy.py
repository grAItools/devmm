"""Host-side staging for `DeviceBuffer`'s byte-level copy helpers.

The helpers stage host bytes in ctypes storage and move them through the
runtime's `memcpy` primitive (design §3.5, §4.1). The ctypes lives here so
the core domain model imports none of it (docs/style.md).
"""

from __future__ import annotations

import ctypes

from devmm._core.stream import Stream
from devmm._runtimes.base import CopyKind, DeviceRuntime


def copy_from_host(
    runtime: DeviceRuntime, dst: int, data: memoryview, kind: CopyKind, stream: Stream
) -> None:
    """Copy the C-contiguous host bytes `data` to pointer `dst`."""
    # from_buffer_copy stages through a fresh ctypes block because ctypes
    # cannot take a zero-copy pointer into a read-only exporter (bytes); it
    # also keeps `memcpy` a pointer-only primitive.
    staged = (ctypes.c_char * data.nbytes).from_buffer_copy(data)
    runtime.memcpy(dst, ctypes.addressof(staged), data.nbytes, kind, stream)


def copy_to_host(
    runtime: DeviceRuntime, src: int, nbytes: int, kind: CopyKind, stream: Stream
) -> bytes:
    """Read `nbytes` from pointer `src` back as host bytes."""
    staged = (ctypes.c_char * nbytes)()
    runtime.memcpy(ctypes.addressof(staged), src, nbytes, kind, stream)
    return staged.raw
