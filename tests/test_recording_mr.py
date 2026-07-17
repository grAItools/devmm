"""`RecordingMemoryResource` self-tests: deterministic fake pointers, a
complete call log, and misuse detection (double-free, foreign-free,
size-mismatch). This is the allocator test double the rest of the suite
leans on (design §9), so it is verified before anything is built on it.
"""

from __future__ import annotations

import itertools

import pytest

from devmm import Device
from devmm._core.stream import CpuStream
from devmm.testing import RecordingMemoryResource, RecordingMisuseError

STREAM = CpuStream()


def test_fake_pointers_are_deterministic() -> None:
    sizes = [0, 1, 17, 256, 4096, 3]
    first = RecordingMemoryResource()
    second = RecordingMemoryResource()
    assert [first.allocate(n, STREAM) for n in sizes] == [second.allocate(n, STREAM) for n in sizes]


def test_fake_pointers_are_nonzero_aligned_and_disjoint() -> None:
    mr = RecordingMemoryResource(guaranteed_alignment=64)
    spans: list[tuple[int, int]] = []
    for nbytes in (0, 1, 63, 64, 65, 1000):
        ptr = mr.allocate(nbytes, STREAM)
        assert ptr != 0
        assert ptr % 64 == 0
        spans.append((ptr, ptr + max(nbytes, 1)))
    spans.sort()
    for (_, end), (start, _) in itertools.pairwise(spans):
        assert end <= start


def test_zero_byte_allocations_get_distinct_pointers_and_free_round_trips() -> None:
    mr = RecordingMemoryResource()
    first = mr.allocate(0, STREAM)
    second = mr.allocate(0, STREAM)
    assert first != second
    mr.deallocate(first, 0, STREAM)
    mr.deallocate(second, 0, STREAM)
    assert mr.live == {}


def test_every_call_is_logged_in_order_with_exact_arguments() -> None:
    mr = RecordingMemoryResource()
    alloc_stream, dealloc_stream = CpuStream(), CpuStream()
    ptr = mr.allocate(16, alloc_stream)
    mr.deallocate(ptr, 16, dealloc_stream)
    assert mr.calls == [
        ("allocate", ptr, 16, alloc_stream),
        ("deallocate", ptr, 16, dealloc_stream),
    ]
    assert mr.calls[0][3] is alloc_stream
    assert mr.calls[1][3] is dealloc_stream


def test_double_free_raises() -> None:
    mr = RecordingMemoryResource()
    ptr = mr.allocate(8, STREAM)
    mr.deallocate(ptr, 8, STREAM)
    with pytest.raises(RecordingMisuseError, match="double-free"):
        mr.deallocate(ptr, 8, STREAM)


def test_foreign_free_raises() -> None:
    mr = RecordingMemoryResource()
    with pytest.raises(RecordingMisuseError, match="never allocated"):
        mr.deallocate(0xBEEF, 8, STREAM)


def test_rejected_deallocate_is_still_logged() -> None:
    mr = RecordingMemoryResource()
    with pytest.raises(RecordingMisuseError):
        mr.deallocate(0xBEEF, 8, STREAM)
    assert mr.calls == [("deallocate", 0xBEEF, 8, STREAM)]


def test_size_mismatch_raises_and_keeps_the_allocation_live() -> None:
    mr = RecordingMemoryResource()
    ptr = mr.allocate(32, STREAM)
    with pytest.raises(RecordingMisuseError, match="32"):
        mr.deallocate(ptr, 16, STREAM)
    assert mr.live == {ptr: 32}
    mr.deallocate(ptr, 32, STREAM)
    assert mr.live == {}


def test_negative_allocation_size_is_rejected() -> None:
    with pytest.raises(ValueError):
        RecordingMemoryResource().allocate(-1, STREAM)


def test_non_positive_guaranteed_alignment_is_rejected() -> None:
    with pytest.raises(ValueError):
        RecordingMemoryResource(guaranteed_alignment=0)


def test_live_tracks_outstanding_allocations() -> None:
    mr = RecordingMemoryResource()
    first = mr.allocate(10, STREAM)
    second = mr.allocate(20, STREAM)
    assert mr.live == {first: 10, second: 20}
    mr.deallocate(first, 10, STREAM)
    assert mr.live == {second: 20}


def test_capability_probes_are_configurable() -> None:
    default = RecordingMemoryResource()
    assert default.stream_ordered is True
    assert default.guaranteed_alignment() == 256
    assert default.available_memory() is None
    configured = RecordingMemoryResource(stream_ordered=False, guaranteed_alignment=32)
    assert configured.stream_ordered is False
    assert configured.guaranteed_alignment() == 32


def test_device_defaults_to_cpu_and_is_configurable() -> None:
    assert RecordingMemoryResource().device == Device.from_string("cpu")
    cuda = Device.from_string("cuda:1")
    assert RecordingMemoryResource(cuda).device is cuda


def test_recording_mr_fixture_provides_a_fresh_instance(
    recording_mr: RecordingMemoryResource,
) -> None:
    assert isinstance(recording_mr, RecordingMemoryResource)
    assert recording_mr.calls == []
    assert recording_mr.live == {}
