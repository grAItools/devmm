"""Pytest mixin over the memory-resource conformance checks in
`devmm.testing._conformance` (design §9).

Mix `MemoryResourceConformance` into a pytest test class and implement
`make_mr()`; each inherited test method runs one contract check, so the
allocate/deallocate contract shows up as individually named test IDs. The
functional entry point over the same checks is the public
`devmm.testing.mr_conformance`.

The checks dereference raw pointers with `ctypes`, so this mixin is only
valid for memory the host can address (CPU MRs, pinned host memory). pytest
and hypothesis are test-time tools, not runtime dependencies of devmm: this
module is importable only inside a test environment and is deliberately not
re-exported from `devmm.testing`.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from devmm._core.memory_resource import DeviceMemoryResource
from devmm._core.stream import CpuStream, Stream
from devmm.testing import _conformance


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
        _conformance.check_write_then_read(
            self.make_mr(), self.make_stream(), _conformance.host_write, _conformance.host_read
        )

    def test_live_allocations_do_not_alias(self) -> None:
        _conformance.check_no_aliasing(
            self.make_mr(), self.make_stream(), _conformance.host_write, _conformance.host_read
        )

    def test_guaranteed_alignment_is_honest(self) -> None:
        _conformance.check_guaranteed_alignment(self.make_mr(), self.make_stream())

    def test_requested_alignment_is_delivered(self) -> None:
        # The hypothesis-decorated function is created per invocation: a
        # @given method inherited by several conformance subclasses would be
        # shared across executors, which hypothesis rejects as flaky
        # (HealthCheck.differing_executors).
        @settings(deadline=None)
        @given(
            nbytes=st.integers(min_value=1, max_value=1 << 12),
            alignment=st.sampled_from(_conformance._ALIGNMENTS),
        )
        def check(nbytes: int, alignment: int) -> None:
            _conformance.check_requested_alignment(
                self.make_mr(alignment=alignment),
                self.make_stream(),
                _conformance.host_write,
                _conformance.host_read,
                nbytes=nbytes,
                alignment=alignment,
            )

        check()

    def test_bookkeeping_is_empty_after_alloc_free_pairs(self) -> None:
        _conformance.check_bookkeeping(self.make_mr(), self.make_stream())

    def test_double_free_raises(self) -> None:
        _conformance.check_double_free(self.make_mr(), self.make_stream())

    def test_freeing_an_unknown_pointer_raises(self) -> None:
        _conformance.check_foreign_free(self.make_mr(), self.make_stream())

    def test_zero_byte_allocation_round_trips(self) -> None:
        _conformance.check_zero_byte(self.make_mr(), self.make_stream())

    def test_negative_allocation_size_is_rejected(self) -> None:
        _conformance.check_negative_size(self.make_mr(), self.make_stream())
