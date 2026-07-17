"""The thin `Stream` abstraction (opaque handle + ordering primitives), the
CPU no-op stream, and the `DEFAULT`/`LEGACY_DEFAULT`/`PER_THREAD_DEFAULT`
sentinels (design §3.2).
"""

from __future__ import annotations

import abc
from collections.abc import Callable
from typing import final

from devmm._core.device import Device, DeviceType

_CPU_DEVICE = Device(DeviceType.CPU)


class Stream(abc.ABC):
    """An opaque native stream handle plus ordering primitives.

    Deliberately thin — the library never launches kernels, it only needs to
    order (de)allocations and DLPack handoffs (design §3.2).
    """

    device: Device

    @property
    @abc.abstractmethod
    def handle(self) -> int:
        """The native handle (cudaStream_t / hipStream_t) as an int."""

    @abc.abstractmethod
    def synchronize(self) -> None:
        """Block until all work enqueued on this stream has completed."""

    @abc.abstractmethod
    def wait_raw(self, other_handle: int) -> None:
        """Make `self` wait on work currently enqueued on a raw foreign
        stream handle (event record + stream-wait-event). Used for the
        `__dlpack__(stream=...)` consumer handoff."""

    def __cuda_stream__(self) -> tuple[int, int]:
        # cuda.core stream protocol: (version, handle).
        return (0, self.handle)


class CpuStream(Stream):
    """The CPU's single no-op stream.

    Host work is synchronous, so ordering primitives do nothing; having a
    real `Stream` keeps every code path uniform without
    `if device.type == CPU` branches in the core (design §3.2).
    """

    def __init__(self, device: Device = _CPU_DEVICE) -> None:
        if device.type is not DeviceType.CPU:
            raise ValueError(f"CpuStream requires a cpu device, got {device}")
        self.device = device

    @property
    def handle(self) -> int:
        return 0

    def synchronize(self) -> None:
        return None

    def wait_raw(self, other_handle: int) -> None:
        return None


@final
class StreamSentinel:
    """Placeholder naming a platform default stream (design §3.2).

    Not a `Stream`: each runtime maps a sentinel to its own magic native
    handle when the sentinel reaches it. Identity is the whole semantics —
    copying, deep-copying and pickling all resolve back to the module-level
    singleton, so sentinels are comparable with `is` everywhere.
    """

    __slots__ = ("_name",)

    def __init__(self, name: str) -> None:
        self._name = name

    def __repr__(self) -> str:
        return f"devmm.{self._name}"

    def __copy__(self) -> StreamSentinel:
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> StreamSentinel:
        return self

    def __reduce__(self) -> tuple[Callable[[str], StreamSentinel], tuple[str]]:
        return (_sentinel_by_name, (self._name,))


DEFAULT = StreamSentinel("DEFAULT")
LEGACY_DEFAULT = StreamSentinel("LEGACY_DEFAULT")
PER_THREAD_DEFAULT = StreamSentinel("PER_THREAD_DEFAULT")

_SENTINELS = {
    sentinel._name: sentinel for sentinel in (DEFAULT, LEGACY_DEFAULT, PER_THREAD_DEFAULT)
}


def _sentinel_by_name(name: str) -> StreamSentinel:
    """Unpickling hook: resolve a sentinel back to its module singleton."""
    return _SENTINELS[name]
