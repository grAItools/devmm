"""Shared plumbing for the provide-direction integrations (design §6): the
`Installer` handle every `install()` returns, the direct consume+provide
cycle guard, and the stream shims that carry third-party stream handles into
devmm MR calls.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from types import TracebackType

from devmm._core.device import Device
from devmm._core.memory_resource import DeviceMemoryResource
from devmm._core.stream import Stream
from devmm._runtimes._discovery import runtime_for

__all__ = ["Installer"]


class Installer:
    """Reversible handle to one third-party installation (design §6).

    `uninstall()` restores the hook state captured at install time. It is
    idempotent — after the first call (or context exit) the handle is spent,
    so a stale handle can never clobber somebody else's newer installation.
    The context-manager form restores on exit, normal or exceptional.
    Stacked installs must be uninstalled in LIFO order: a non-LIFO
    uninstall restores *its* captured prior state and thereby overwrites
    any installation made after it.
    """

    def __init__(self, library: str, restore: Callable[[], None]) -> None:
        self._library = library
        self._restore = restore
        self._lock = threading.Lock()
        self._installed = True

    @property
    def installed(self) -> bool:
        """False once this installation has been reverted."""
        return self._installed

    def uninstall(self) -> None:
        """Restore the state captured at install time (first call only)."""
        # Swap the flag under a lock so concurrent uninstalls cannot both
        # observe "installed" and run the restore twice.
        with self._lock:
            if not self._installed:
                return
            self._installed = False
        self._restore()

    def __enter__(self) -> Installer:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.uninstall()

    def __repr__(self) -> str:
        state = "installed" if self._installed else "uninstalled"
        return f"{type(self).__name__}(library={self._library!r}, {state})"


def ensure_no_cycle(
    mr: DeviceMemoryResource, forbidden: type[DeviceMemoryResource], library: str
) -> None:
    """Refuse the direct consume+provide composition (design §6).

    Installing an MR whose chain already consumes `library`'s allocator
    would make that allocator (indirectly) call itself; the walk follows the
    adaptor `upstream` chain, which is where the direct case lives.
    """
    current: DeviceMemoryResource | None = mr
    while current is not None:
        if isinstance(current, forbidden):
            raise ValueError(
                f"installing {mr!r} into {library} would create an allocation "
                f"cycle: its resource chain contains {type(current).__name__}, "
                f"which already allocates from {library} (design §6)"
            )
        current = getattr(current, "upstream", None)


class ForeignHandleStream(Stream):
    """A third-party library's raw stream handle, carried into MR calls.

    The provide-direction hooks receive foreign streams (CuPy's current
    stream, the stream rmm hands its callbacks) that must reach
    `mr.allocate(nbytes, stream)` as a devmm `Stream`. Allocation paths read
    only `handle`/`__cuda_stream__`; the ordering primitives resolve the
    device runtime on demand and raise `RuntimeUnavailableError` where no
    runtime serves the device.
    """

    def __init__(self, device: Device, handle: int) -> None:
        if handle < 0:
            raise ValueError(f"stream handles are non-negative ints, got {handle}")
        self.device = device
        self._handle = handle

    @property
    def handle(self) -> int:
        return self._handle

    def synchronize(self) -> None:
        self._native().synchronize()

    def wait_raw(self, other_handle: int) -> None:
        self._native().wait_raw(other_handle)

    def _native(self) -> Stream:
        return runtime_for(self.device).wrap_stream(self.device, self._handle)


def stream_handle_of(obj: object) -> int:
    """Best-effort raw handle of a foreign stream object.

    None means the default stream (0); ints pass through; then the CUDA
    stream protocol; then the attribute spellings the ecosystem uses
    (devmm/numba `handle`, CuPy `ptr`, rmm `value`).
    """
    if obj is None:
        return 0
    if isinstance(obj, int):
        return obj
    protocol = getattr(obj, "__cuda_stream__", None)
    if protocol is not None:
        _, handle = protocol()
        if isinstance(handle, int):
            return handle
        raise TypeError(f"__cuda_stream__ must yield an int handle, got {handle!r}")
    for attribute in ("handle", "ptr", "value"):
        value = getattr(obj, attribute, None)
        if isinstance(value, int):
            return value
    raise TypeError(
        f"cannot read a stream handle from {type(obj).__name__!r}: expected None, "
        "an int, an object exposing __cuda_stream__, or one with an int "
        "handle/ptr/value attribute"
    )
