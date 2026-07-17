"""`RecordingMemoryResource`: the allocator test double the suite lives on.

It hands out deterministic fake pointers (no real memory), logs every call
verbatim so tests can assert exact `(nbytes, stream)` forwarding and
stream-ordering contracts, and turns caller misuse — double-free,
foreign-free, size-mismatch — into `RecordingMisuseError` (design §9).
"""

from __future__ import annotations

import threading

from devmm._core.device import Device, DeviceType
from devmm._core.memory_resource import DeviceMemoryResource
from devmm._core.stream import Stream

_CPU_DEVICE = Device(DeviceType.CPU)

# ("allocate" | "deallocate", ptr, nbytes, stream), in call order.
RecordedCall = tuple[str, int, int, Stream]


class RecordingMisuseError(AssertionError):
    """A caller violated the allocate/deallocate contract.

    Subclasses `AssertionError` so misuse surfaces as a plain test failure
    in suites built on the recording fixture.
    """


class RecordingMemoryResource(DeviceMemoryResource):
    """A `DeviceMemoryResource` that fakes allocation for tests.

    Pointers are deterministic — a cursor walks upward from
    `guaranteed_alignment`, aligning each result and never reusing an
    address — so two instances fed the same call sequence return the same
    pointers, spans never overlap, and 0 (the null pointer) is never handed
    out.
    """

    def __init__(
        self,
        device: Device = _CPU_DEVICE,
        *,
        stream_ordered: bool = True,
        guaranteed_alignment: int = 256,
    ) -> None:
        if guaranteed_alignment < 1:
            raise ValueError(f"guaranteed_alignment must be >= 1, got {guaranteed_alignment}")
        self.device = device
        self._stream_ordered = stream_ordered
        self._alignment = guaranteed_alignment
        self._lock = threading.Lock()
        self._cursor = guaranteed_alignment
        self._live: dict[int, int] = {}
        self._freed: set[int] = set()
        self.calls: list[RecordedCall] = []

    @property
    def stream_ordered(self) -> bool:
        return self._stream_ordered

    def guaranteed_alignment(self) -> int:
        return self._alignment

    @property
    def live(self) -> dict[int, int]:
        """Outstanding allocations as a `ptr -> nbytes` snapshot."""
        with self._lock:
            return dict(self._live)

    def allocate(self, nbytes: int, stream: Stream) -> int:
        if nbytes < 0:
            raise ValueError(f"cannot allocate a negative size ({nbytes} bytes)")
        with self._lock:
            remainder = self._cursor % self._alignment
            if remainder:
                self._cursor += self._alignment - remainder
            ptr = self._cursor
            # Advance at least one byte so zero-byte allocations still get
            # distinct pointers.
            self._cursor += max(nbytes, 1)
            self._live[ptr] = nbytes
            self.calls.append(("allocate", ptr, nbytes, stream))
        return ptr

    def deallocate(self, ptr: int, nbytes: int, stream: Stream) -> None:
        with self._lock:
            # Rejected calls are logged too: the log is the record of what
            # the code under test *asked for*, not of what was valid.
            self.calls.append(("deallocate", ptr, nbytes, stream))
            if ptr not in self._live:
                if ptr in self._freed:
                    raise RecordingMisuseError(f"double-free of pointer {ptr:#x} on {self.device}")
                raise RecordingMisuseError(
                    f"foreign-free: pointer {ptr:#x} was never allocated by {self!r}"
                )
            expected = self._live[ptr]
            if nbytes != expected:
                raise RecordingMisuseError(
                    f"size-mismatch freeing pointer {ptr:#x}: allocated {expected} bytes, "
                    f"deallocate got {nbytes}"
                )
            del self._live[ptr]
            self._freed.add(ptr)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(device={self.device})"
