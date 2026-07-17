"""`DeviceMemoryResource` ABC (rmm-isomorphic allocate/deallocate + capability
probes) and the layer's own adaptors: Statistics, Logging, Limiting, Callback
(design §3.3).
"""

from __future__ import annotations

import abc
import logging
import threading
from collections.abc import Callable

from devmm._core.device import Device
from devmm._core.stream import Stream

_LOGGER = logging.getLogger("devmm.mr")


class DeviceMemoryResource(abc.ABC):
    """The central allocation abstraction, deliberately isomorphic to rmm's
    `allocate(nbytes, stream)` / `deallocate(ptr, nbytes, stream)` so the
    mental model (and rmm's docs) transfer (design §3.3)."""

    device: Device

    @abc.abstractmethod
    def allocate(self, nbytes: int, stream: Stream) -> int:
        """Return a device pointer (int). Raise `MemoryError` on failure.

        Stream-ordered contract: the memory is usable on `stream`
        immediately, elsewhere only after synchronization.
        """

    @abc.abstractmethod
    def deallocate(self, ptr: int, nbytes: int, stream: Stream) -> None:
        """Release `ptr` (as returned by `allocate`), ordered on `stream`."""

    @property
    def stream_ordered(self) -> bool:
        """True if (de)allocation is genuinely stream-ordered.

        Synchronous MRs (malloc, cudaMalloc, ...) return False; they must
        still be safe under the stream-ordered calling convention:
        `allocate()` may ignore `stream`; `deallocate()` must not release
        memory that could still be in use on any stream (design §3.3).
        """
        return False

    def guaranteed_alignment(self) -> int:
        """Minimum alignment (bytes) of every returned pointer; 1 if unknown."""
        return 1

    def available_memory(self) -> tuple[int, int] | None:
        """`(free, total)` bytes if the resource can tell, else None."""
        return None


class CallbackMemoryResource(DeviceMemoryResource):
    """Pure-Python escape hatch: delegate to user callables (design §3.3).

    `alloc_fn(nbytes, stream) -> ptr` and `dealloc_fn(ptr, nbytes, stream)`
    are invoked with the caller's exact arguments; exceptions they raise
    propagate untouched.
    """

    def __init__(
        self,
        alloc_fn: Callable[[int, Stream], int],
        dealloc_fn: Callable[[int, int, Stream], None],
        device: Device,
    ) -> None:
        self.device = device
        self._alloc_fn = alloc_fn
        self._dealloc_fn = dealloc_fn

    def allocate(self, nbytes: int, stream: Stream) -> int:
        return self._alloc_fn(nbytes, stream)

    def deallocate(self, ptr: int, nbytes: int, stream: Stream) -> None:
        self._dealloc_fn(ptr, nbytes, stream)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(device={self.device})"


class _ForwardingAdaptor(DeviceMemoryResource):
    """Shared adaptor plumbing.

    `upstream` is a strong reference by design: the adaptor chain must keep
    the wrapped allocator alive for as long as any allocation can reach it —
    the lifetime hazard rmm documents with its non-owning resource refs,
    fixed at this layer (design §3.3). Capability probes and `device`
    delegate to `upstream` because an adaptor changes accounting, never
    allocation semantics.
    """

    upstream: DeviceMemoryResource

    def __init__(self, upstream: DeviceMemoryResource) -> None:
        self.upstream = upstream
        self.device = upstream.device

    def allocate(self, nbytes: int, stream: Stream) -> int:
        return self.upstream.allocate(nbytes, stream)

    def deallocate(self, ptr: int, nbytes: int, stream: Stream) -> None:
        self.upstream.deallocate(ptr, nbytes, stream)

    @property
    def stream_ordered(self) -> bool:
        return self.upstream.stream_ordered

    def guaranteed_alignment(self) -> int:
        return self.upstream.guaranteed_alignment()

    def available_memory(self) -> tuple[int, int] | None:
        return self.upstream.available_memory()

    def __repr__(self) -> str:
        return f"{type(self).__name__}(upstream={self.upstream!r})"


class StatisticsAdaptor(_ForwardingAdaptor):
    """Byte accounting over an upstream MR.

    Invariants: `current_bytes` equals the sum of live allocation sizes,
    `peak_bytes` is the maximum `current_bytes` ever observed, and
    `total_bytes` grows monotonically. Only *successful* upstream calls are
    counted, and a mutating adaptor is lock-protected (design §8).
    """

    def __init__(self, upstream: DeviceMemoryResource) -> None:
        super().__init__(upstream)
        self._lock = threading.Lock()
        self._current = 0
        self._peak = 0
        self._total = 0

    def allocate(self, nbytes: int, stream: Stream) -> int:
        ptr = self.upstream.allocate(nbytes, stream)
        with self._lock:
            self._current += nbytes
            self._peak = max(self._peak, self._current)
            self._total += nbytes
        return ptr

    def deallocate(self, ptr: int, nbytes: int, stream: Stream) -> None:
        self.upstream.deallocate(ptr, nbytes, stream)
        with self._lock:
            self._current -= nbytes

    @property
    def current_bytes(self) -> int:
        """Bytes currently allocated (sum over live allocations)."""
        with self._lock:
            return self._current

    @property
    def peak_bytes(self) -> int:
        """High-water mark of `current_bytes`."""
        with self._lock:
            return self._peak

    @property
    def total_bytes(self) -> int:
        """Bytes ever allocated (monotone; never decreases on deallocate)."""
        with self._lock:
            return self._total


class LoggingAdaptor(_ForwardingAdaptor):
    """Log every successful (de)allocation through an upstream MR.

    Records go to the `devmm.mr` logger by default (design §8); pass
    `logger` to redirect them.
    """

    def __init__(
        self, upstream: DeviceMemoryResource, logger: logging.Logger | None = None
    ) -> None:
        super().__init__(upstream)
        self._logger = _LOGGER if logger is None else logger

    def allocate(self, nbytes: int, stream: Stream) -> int:
        ptr = self.upstream.allocate(nbytes, stream)
        self._logger.info(
            "allocate device=%s nbytes=%d ptr=%#x stream=%r", self.device, nbytes, ptr, stream
        )
        return ptr

    def deallocate(self, ptr: int, nbytes: int, stream: Stream) -> None:
        self.upstream.deallocate(ptr, nbytes, stream)
        self._logger.info(
            "deallocate device=%s nbytes=%d ptr=%#x stream=%r", self.device, nbytes, ptr, stream
        )


class LimitingAdaptor(_ForwardingAdaptor):
    """Cap the bytes concurrently allocated through an upstream MR.

    Boundary-exact: an allocation that lands exactly on `limit_bytes`
    succeeds; one byte more raises `MemoryError`. Failed allocations —
    whether refused here or failed upstream — leave the budget untouched.
    """

    def __init__(self, upstream: DeviceMemoryResource, limit_bytes: int) -> None:
        if limit_bytes < 0:
            raise ValueError(f"limit_bytes must be non-negative, got {limit_bytes}")
        super().__init__(upstream)
        self._lock = threading.Lock()
        self._limit = limit_bytes
        self._used = 0

    def allocate(self, nbytes: int, stream: Stream) -> int:
        # Reserve before calling upstream so concurrent allocations cannot
        # jointly oversubscribe the limit; roll back if upstream fails.
        with self._lock:
            if self._used + nbytes > self._limit:
                raise MemoryError(
                    f"allocation of {nbytes} bytes on {self.device} would exceed the "
                    f"{self._limit}-byte limit ({self._used} bytes in use) in {self!r}"
                )
            self._used += nbytes
        try:
            return self.upstream.allocate(nbytes, stream)
        except BaseException:
            with self._lock:
                self._used -= nbytes
            raise

    def deallocate(self, ptr: int, nbytes: int, stream: Stream) -> None:
        self.upstream.deallocate(ptr, nbytes, stream)
        with self._lock:
            self._used -= nbytes

    def __repr__(self) -> str:
        return f"{type(self).__name__}(limit_bytes={self._limit}, upstream={self.upstream!r})"
