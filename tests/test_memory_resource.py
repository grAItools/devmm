"""Memory-resource contract tests: ABC capability defaults, adaptor
forwarding + strong-upstream lifetime, `StatisticsAdaptor` accounting
invariants (hypothesis interleavings + thread soak), `LimitingAdaptor`
boundary exactness, `LoggingAdaptor` records, and `CallbackMemoryResource`
pass-through (design §3.3).
"""

from __future__ import annotations

import gc
import logging
import random
import threading
import weakref
from collections.abc import Callable

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from devmm import (
    CallbackMemoryResource,
    Device,
    DeviceMemoryResource,
    LimitingAdaptor,
    LoggingAdaptor,
    StatisticsAdaptor,
    Stream,
)
from devmm._core.stream import CpuStream
from devmm.testing import RecordingMemoryResource

STREAM = CpuStream()
CPU = Device.from_string("cpu")

# The union (not the ABC) keeps `.upstream` accesses statically typed.
Adaptor = StatisticsAdaptor | LoggingAdaptor | LimitingAdaptor
AdaptorFactory = Callable[[DeviceMemoryResource], Adaptor]

# Every adaptor must be a transparent proxy for allocation semantics; the
# suite below runs the shared contracts over all of them.
ADAPTORS: tuple[AdaptorFactory, ...] = (
    StatisticsAdaptor,
    LoggingAdaptor,
    lambda upstream: LimitingAdaptor(upstream, limit_bytes=1 << 40),
)
ADAPTOR_IDS = ("statistics", "logging", "limiting")


class _ProbeMemoryResource(DeviceMemoryResource):
    """Minimal concrete MR with distinctive capability answers, so tests can
    tell delegation from the ABC defaults."""

    def __init__(self) -> None:
        self.device = Device.from_string("cuda:1")

    def allocate(self, nbytes: int, stream: Stream) -> int:
        return 0xA110C

    def deallocate(self, ptr: int, nbytes: int, stream: Stream) -> None:
        pass

    @property
    def stream_ordered(self) -> bool:
        return True

    def guaranteed_alignment(self) -> int:
        return 128

    def available_memory(self) -> tuple[int, int] | None:
        return (123, 456)


class _DefaultsMemoryResource(DeviceMemoryResource):
    """Concrete MR that overrides nothing optional: pins the ABC defaults."""

    def __init__(self) -> None:
        self.device = CPU

    def allocate(self, nbytes: int, stream: Stream) -> int:
        return 1

    def deallocate(self, ptr: int, nbytes: int, stream: Stream) -> None:
        pass


def test_abc_requires_exactly_allocate_and_deallocate() -> None:
    assert DeviceMemoryResource.__abstractmethods__ == frozenset({"allocate", "deallocate"})


def test_abc_capability_defaults() -> None:
    mr = _DefaultsMemoryResource()
    assert mr.stream_ordered is False
    assert mr.guaranteed_alignment() == 1
    assert mr.available_memory() is None


@pytest.mark.parametrize("factory", ADAPTORS, ids=ADAPTOR_IDS)
def test_adaptor_forwards_nbytes_and_stream_exactly(factory: AdaptorFactory) -> None:
    upstream = RecordingMemoryResource()
    adaptor = factory(upstream)
    alloc_stream, dealloc_stream = CpuStream(), CpuStream()
    ptr = adaptor.allocate(1234, alloc_stream)
    adaptor.deallocate(ptr, 1234, dealloc_stream)
    assert upstream.calls == [
        ("allocate", ptr, 1234, alloc_stream),
        ("deallocate", ptr, 1234, dealloc_stream),
    ]
    assert upstream.calls[0][3] is alloc_stream
    assert upstream.calls[1][3] is dealloc_stream


@pytest.mark.parametrize("factory", ADAPTORS, ids=ADAPTOR_IDS)
def test_adaptor_exposes_and_delegates_to_upstream(factory: AdaptorFactory) -> None:
    upstream = _ProbeMemoryResource()
    adaptor = factory(upstream)
    assert adaptor.upstream is upstream
    assert adaptor.device is upstream.device
    assert adaptor.stream_ordered is True
    assert adaptor.guaranteed_alignment() == 128
    assert adaptor.available_memory() == (123, 456)


def test_adaptor_chain_keeps_upstream_alive_through_gc() -> None:
    upstream = RecordingMemoryResource()
    probe = weakref.ref(upstream)
    chain = StatisticsAdaptor(LoggingAdaptor(LimitingAdaptor(upstream, limit_bytes=1 << 20)))
    del upstream
    gc.collect()
    alive = probe()
    assert alive is not None
    ptr = chain.allocate(64, STREAM)
    assert alive.calls[-1] == ("allocate", ptr, 64, STREAM)
    del alive
    del chain
    gc.collect()
    assert probe() is None


@st.composite
def interleavings(draw: st.DrawFn) -> list[tuple[str, int]]:
    """Valid alloc/free interleavings.

    `("alloc", nbytes)` allocates; `("free", serial)` frees the allocation
    made by the serial-th alloc op (0-based), which is guaranteed live.
    """
    ops: list[tuple[str, int]] = []
    live: list[int] = []
    serial = 0
    for _ in range(draw(st.integers(min_value=1, max_value=50))):
        free_next = draw(st.booleans()) if live else False
        if free_next:
            index = draw(st.integers(0, len(live) - 1))
            ops.append(("free", live.pop(index)))
        else:
            ops.append(("alloc", draw(st.integers(0, 1 << 16))))
            live.append(serial)
            serial += 1
    return ops


@settings(deadline=None)
@given(interleavings())
def test_statistics_invariants_under_interleaved_ops(ops: list[tuple[str, int]]) -> None:
    stats = StatisticsAdaptor(RecordingMemoryResource())
    live: dict[int, tuple[int, int]] = {}
    serial = 0
    expected_total = 0
    expected_peak = 0
    for op, value in ops:
        if op == "alloc":
            ptr = stats.allocate(value, STREAM)
            live[serial] = (ptr, value)
            serial += 1
            expected_total += value
        else:
            ptr, nbytes = live.pop(value)
            stats.deallocate(ptr, nbytes, STREAM)
        expected_current = sum(nbytes for _, nbytes in live.values())
        expected_peak = max(expected_peak, expected_current)
        assert stats.current_bytes == expected_current
        assert stats.total_bytes == expected_total
        assert stats.peak_bytes == expected_peak


def test_statistics_thread_soak_ends_balanced() -> None:
    upstream = RecordingMemoryResource()
    stats = StatisticsAdaptor(upstream)
    n_threads, pairs = 8, 500
    totals = [0] * n_threads
    errors: list[BaseException] = []

    def worker(slot: int) -> None:
        rng = random.Random(slot)
        stream = CpuStream()
        try:
            for _ in range(pairs):
                nbytes = rng.randrange(1, 4096)
                ptr = stats.allocate(nbytes, stream)
                totals[slot] += nbytes
                stats.deallocate(ptr, nbytes, stream)
        except BaseException as exc:
            # A raise in a worker thread would otherwise vanish; the main
            # thread asserts `errors == []` after joining.
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(slot,)) for slot in range(n_threads)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert stats.current_bytes == 0
    assert stats.total_bytes == sum(totals)
    assert 0 < stats.peak_bytes <= n_threads * 4095
    assert upstream.live == {}


def test_statistics_does_not_count_failed_allocations() -> None:
    def failing_alloc(nbytes: int, stream: Stream) -> int:
        raise MemoryError("injected upstream failure")

    upstream = CallbackMemoryResource(failing_alloc, lambda ptr, nbytes, stream: None, CPU)
    stats = StatisticsAdaptor(upstream)
    with pytest.raises(MemoryError):
        stats.allocate(64, STREAM)
    assert stats.current_bytes == 0
    assert stats.total_bytes == 0
    assert stats.peak_bytes == 0


def test_limiting_on_limit_allocation_succeeds() -> None:
    upstream = RecordingMemoryResource()
    limited = LimitingAdaptor(upstream, limit_bytes=64)
    ptr = limited.allocate(64, STREAM)
    assert upstream.live == {ptr: 64}


def test_limiting_one_byte_over_raises_memory_error() -> None:
    limited = LimitingAdaptor(RecordingMemoryResource(), limit_bytes=64)
    with pytest.raises(MemoryError, match="cpu:0"):
        limited.allocate(65, STREAM)


def test_limiting_failed_allocation_is_uncounted() -> None:
    upstream = RecordingMemoryResource()
    limited = LimitingAdaptor(upstream, limit_bytes=64)
    with pytest.raises(MemoryError):
        limited.allocate(65, STREAM)
    assert upstream.calls == []
    # The full budget is still available after the refusal.
    ptr = limited.allocate(64, STREAM)
    assert upstream.live == {ptr: 64}


def test_limiting_restores_budget_when_upstream_fails() -> None:
    def failing_alloc(nbytes: int, stream: Stream) -> int:
        raise MemoryError("injected upstream failure")

    upstream = CallbackMemoryResource(failing_alloc, lambda ptr, nbytes, stream: None, CPU)
    limited = LimitingAdaptor(upstream, limit_bytes=64)
    with pytest.raises(MemoryError, match="injected"):
        limited.allocate(64, STREAM)
    # "injected" (not the limit message) proves the retry passed the limit
    # check again: the failed attempt left no residue in the budget.
    with pytest.raises(MemoryError, match="injected"):
        limited.allocate(64, STREAM)


def test_limiting_deallocate_releases_budget() -> None:
    upstream = RecordingMemoryResource()
    limited = LimitingAdaptor(upstream, limit_bytes=64)
    ptr = limited.allocate(64, STREAM)
    limited.deallocate(ptr, 64, STREAM)
    second = limited.allocate(64, STREAM)
    assert upstream.live == {second: 64}


def test_limiting_rejects_negative_limit() -> None:
    with pytest.raises(ValueError):
        LimitingAdaptor(RecordingMemoryResource(), limit_bytes=-1)


def test_logging_adaptor_emits_structured_records(caplog: pytest.LogCaptureFixture) -> None:
    adaptor = LoggingAdaptor(RecordingMemoryResource())
    with caplog.at_level(logging.INFO, logger="devmm.mr"):
        ptr = adaptor.allocate(96, STREAM)
        adaptor.deallocate(ptr, 96, STREAM)
    assert [record.name for record in caplog.records] == ["devmm.mr", "devmm.mr"]
    alloc_message, dealloc_message = (record.getMessage() for record in caplog.records)
    assert "allocate" in alloc_message
    assert "nbytes=96" in alloc_message
    assert format(ptr, "#x") in alloc_message
    assert "deallocate" in dealloc_message
    assert "nbytes=96" in dealloc_message


def test_logging_adaptor_accepts_a_custom_logger(caplog: pytest.LogCaptureFixture) -> None:
    logger = logging.getLogger("test.devmm.custom")
    adaptor = LoggingAdaptor(RecordingMemoryResource(), logger=logger)
    with caplog.at_level(logging.INFO, logger="test.devmm.custom"):
        ptr = adaptor.allocate(8, STREAM)
        adaptor.deallocate(ptr, 8, STREAM)
    assert [record.name for record in caplog.records] == ["test.devmm.custom"] * 2


def test_callback_invokes_callbacks_with_exact_arguments() -> None:
    alloc_calls: list[tuple[int, Stream]] = []
    dealloc_calls: list[tuple[int, int, Stream]] = []

    def alloc(nbytes: int, stream: Stream) -> int:
        alloc_calls.append((nbytes, stream))
        return 0xABC0

    def dealloc(ptr: int, nbytes: int, stream: Stream) -> None:
        dealloc_calls.append((ptr, nbytes, stream))

    mr = CallbackMemoryResource(alloc, dealloc, CPU)
    stream = CpuStream()
    assert mr.allocate(48, stream) == 0xABC0
    mr.deallocate(0xABC0, 48, stream)
    assert alloc_calls == [(48, stream)]
    assert alloc_calls[0][1] is stream
    assert dealloc_calls == [(0xABC0, 48, stream)]
    assert mr.device is CPU


class _Boom(Exception):
    pass


def test_callback_propagates_alloc_exceptions_untouched() -> None:
    boom = _Boom("original")

    def alloc(nbytes: int, stream: Stream) -> int:
        raise boom

    mr = CallbackMemoryResource(alloc, lambda ptr, nbytes, stream: None, CPU)
    with pytest.raises(_Boom) as excinfo:
        mr.allocate(1, STREAM)
    assert excinfo.value is boom


def test_callback_propagates_dealloc_exceptions_untouched() -> None:
    boom = _Boom("original")

    def dealloc(ptr: int, nbytes: int, stream: Stream) -> None:
        raise boom

    mr = CallbackMemoryResource(lambda nbytes, stream: 1, dealloc, CPU)
    with pytest.raises(_Boom) as excinfo:
        mr.deallocate(1, 1, STREAM)
    assert excinfo.value is boom
