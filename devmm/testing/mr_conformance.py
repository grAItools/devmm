"""Reusable conformance suite for host-visible `DeviceMemoryResource`
implementations (design §9).

Mix `MemoryResourceConformance` into a pytest test class and implement
`make_mr()`; the inherited tests pin the allocate/deallocate contract:
byte-exact writes/reads through returned pointers, no aliasing between live
allocations, requested-alignment delivery and `guaranteed_alignment()`
honesty, bookkeeping hygiene via the `_debug_live_count()` hook, misuse
detection (double-free, unknown-pointer free), and the zero-byte contract.

The suite dereferences raw pointers with `ctypes`, so it is only valid for
memory the host can address (CPU MRs, pinned host memory). pytest and
hypothesis are test-time tools, not runtime dependencies of devmm: this
module is importable only inside a test environment and is deliberately not
re-exported from `devmm.testing`.
"""

from __future__ import annotations

import ctypes

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from devmm._core.memory_resource import DeviceMemoryResource
from devmm._core.stream import CpuStream, Stream

_ALIGNMENTS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096)


def _pattern(nbytes: int) -> bytes:
    # Period 251 (prime) so the pattern can never line up with power-of-two
    # allocation granularities and mask an addressing bug.
    return bytes(i % 251 for i in range(nbytes))


def _write(ptr: int, data: bytes) -> None:
    ctypes.memmove(ptr, data, len(data))


def _read(ptr: int, nbytes: int) -> bytes:
    return ctypes.string_at(ptr, nbytes)


def _live_count(mr: DeviceMemoryResource) -> int:
    """Read the `_debug_live_count()` testing hook devmm MRs expose."""
    hook = getattr(mr, "_debug_live_count", None)
    assert hook is not None, f"{mr!r} does not expose the _debug_live_count() testing hook"
    count = hook()
    assert isinstance(count, int)
    return count


class MemoryResourceConformance:
    """Inherit in a pytest test class and implement `make_mr` (and, for
    non-CPU devices, `make_stream`) to run the whole suite against an MR."""

    def make_mr(self, *, alignment: int | None = None) -> DeviceMemoryResource:
        """Return a fresh MR under test; `alignment=None` means its default."""
        raise NotImplementedError("conformance subclasses must implement make_mr()")

    def make_stream(self) -> Stream:
        """Return the stream (de)allocations are ordered on."""
        return CpuStream()

    def test_write_then_read_is_byte_exact(self) -> None:
        mr = self.make_mr()
        stream = self.make_stream()
        data = _pattern(1024)
        ptr = mr.allocate(len(data), stream)
        _write(ptr, data)
        assert _read(ptr, len(data)) == data
        mr.deallocate(ptr, len(data), stream)

    def test_live_allocations_do_not_alias(self) -> None:
        mr = self.make_mr()
        stream = self.make_stream()
        first = mr.allocate(256, stream)
        _write(first, b"\xaa" * 256)
        second = mr.allocate(256, stream)
        _write(second, b"\x55" * 256)
        assert _read(first, 256) == b"\xaa" * 256
        mr.deallocate(first, 256, stream)
        assert _read(second, 256) == b"\x55" * 256
        mr.deallocate(second, 256, stream)

    def test_guaranteed_alignment_is_honest(self) -> None:
        mr = self.make_mr()
        stream = self.make_stream()
        alignment = mr.guaranteed_alignment()
        assert alignment >= 1
        for nbytes in (1, 3, 17, 255, 4096):
            ptr = mr.allocate(nbytes, stream)
            assert ptr % alignment == 0
            mr.deallocate(ptr, nbytes, stream)

    def test_requested_alignment_is_delivered(self) -> None:
        # The hypothesis-decorated function is created per invocation: a
        # @given method inherited by several conformance subclasses would be
        # shared across executors, which hypothesis rejects as flaky
        # (HealthCheck.differing_executors).
        @settings(deadline=None)
        @given(
            nbytes=st.integers(min_value=1, max_value=1 << 12),
            alignment=st.sampled_from(_ALIGNMENTS),
        )
        def check(nbytes: int, alignment: int) -> None:
            mr = self.make_mr(alignment=alignment)
            stream = self.make_stream()
            ptr = mr.allocate(nbytes, stream)
            assert ptr % alignment == 0
            # Writing and reading the whole span catches off-by-one
            # over-allocation at the aligned offset.
            data = _pattern(nbytes)
            _write(ptr, data)
            assert _read(ptr, nbytes) == data
            mr.deallocate(ptr, nbytes, stream)

        check()

    def test_bookkeeping_is_empty_after_alloc_free_pairs(self) -> None:
        mr = self.make_mr()
        stream = self.make_stream()
        live = [(mr.allocate(nbytes, stream), nbytes) for nbytes in range(0, 64, 7)]
        assert _live_count(mr) == len(live)
        for ptr, nbytes in live:
            mr.deallocate(ptr, nbytes, stream)
        assert _live_count(mr) == 0

    def test_double_free_raises(self) -> None:
        mr = self.make_mr()
        stream = self.make_stream()
        ptr = mr.allocate(64, stream)
        mr.deallocate(ptr, 64, stream)
        with pytest.raises(ValueError):
            mr.deallocate(ptr, 64, stream)

    def test_freeing_an_unknown_pointer_raises(self) -> None:
        mr = self.make_mr()
        stream = self.make_stream()
        ptr = mr.allocate(64, stream)
        with pytest.raises(ValueError):
            mr.deallocate(ptr + 1, 64, stream)
        # The refused free must not have disturbed the real allocation.
        mr.deallocate(ptr, 64, stream)
        assert _live_count(mr) == 0

    def test_zero_byte_allocation_round_trips(self) -> None:
        mr = self.make_mr()
        stream = self.make_stream()
        first = mr.allocate(0, stream)
        second = mr.allocate(0, stream)
        assert first != 0
        assert second != 0
        # Concurrently live zero-byte allocations must still be distinct
        # pointers, or the caller's bookkeeping (and the MR's own) collides.
        assert first != second
        mr.deallocate(first, 0, stream)
        mr.deallocate(second, 0, stream)
        assert _live_count(mr) == 0

    def test_negative_allocation_size_is_rejected(self) -> None:
        mr = self.make_mr()
        with pytest.raises(ValueError):
            mr.allocate(-1, self.make_stream())
