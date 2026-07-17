"""`NumpyHandlerMemoryResource` against the real NumPy (design §5.1, §9):
allocations go through the NEP-49 handler captured at construction —
verified differentially against a devmm-installed handler in both
directions — plus the sibling-MR misuse contracts and the supported-NumPy
range guard.
"""

from __future__ import annotations

import ctypes
import sys

import pytest

from devmm import Device, StatisticsAdaptor, _nep49
from devmm._core.stream import CpuStream
from devmm._runtimes.base import RuntimeUnavailableError
from devmm.integrations import numpy as integrations_numpy
from devmm.mrs.cpu import MallocMemoryResource, NumpyHandlerMemoryResource

np = pytest.importorskip("numpy")

_STREAM = CpuStream()
_CUDA = Device.from_string("cuda:0")


def _pattern(nbytes: int) -> bytes:
    # Period 251 (prime) so the pattern cannot line up with power-of-two
    # allocation granularities and mask an addressing bug.
    return bytes(i % 251 for i in range(nbytes))


class TestConformance:
    def test_write_then_read_is_byte_exact(self) -> None:
        mr = NumpyHandlerMemoryResource()
        data = _pattern(512)
        ptr = mr.allocate(len(data), _STREAM)
        ctypes.memmove(ptr, data, len(data))
        assert ctypes.string_at(ptr, len(data)) == data
        mr.deallocate(ptr, len(data), _STREAM)

    def test_live_allocations_do_not_alias(self) -> None:
        mr = NumpyHandlerMemoryResource()
        first = mr.allocate(256, _STREAM)
        ctypes.memmove(first, b"\xaa" * 256, 256)
        second = mr.allocate(256, _STREAM)
        ctypes.memmove(second, b"\x55" * 256, 256)
        assert ctypes.string_at(first, 256) == b"\xaa" * 256
        mr.deallocate(first, 256, _STREAM)
        assert ctypes.string_at(second, 256) == b"\x55" * 256
        mr.deallocate(second, 256, _STREAM)

    def test_bookkeeping_is_empty_after_alloc_free_pairs(self) -> None:
        mr = NumpyHandlerMemoryResource()
        live = [(mr.allocate(nbytes, _STREAM), nbytes) for nbytes in range(0, 64, 7)]
        assert mr._debug_live_count() == len(live)
        for ptr, nbytes in live:
            mr.deallocate(ptr, nbytes, _STREAM)
        assert mr._debug_live_count() == 0

    def test_double_free_raises(self) -> None:
        mr = NumpyHandlerMemoryResource()
        ptr = mr.allocate(64, _STREAM)
        mr.deallocate(ptr, 64, _STREAM)
        with pytest.raises(ValueError, match="live allocation"):
            mr.deallocate(ptr, 64, _STREAM)

    def test_freeing_an_unknown_pointer_raises(self) -> None:
        mr = NumpyHandlerMemoryResource()
        ptr = mr.allocate(64, _STREAM)
        with pytest.raises(ValueError, match="live allocation"):
            mr.deallocate(ptr + 1, 64, _STREAM)
        mr.deallocate(ptr, 64, _STREAM)

    def test_size_mismatched_free_raises_and_keeps_the_allocation_live(self) -> None:
        mr = NumpyHandlerMemoryResource()
        ptr = mr.allocate(64, _STREAM)
        with pytest.raises(ValueError, match="size mismatch"):
            mr.deallocate(ptr, 63, _STREAM)
        assert mr._debug_live_count() == 1
        mr.deallocate(ptr, 64, _STREAM)

    def test_zero_byte_allocation_round_trips(self) -> None:
        mr = NumpyHandlerMemoryResource()
        first = mr.allocate(0, _STREAM)
        second = mr.allocate(0, _STREAM)
        assert first != 0
        assert second != 0
        assert first != second
        mr.deallocate(first, 0, _STREAM)
        mr.deallocate(second, 0, _STREAM)
        assert mr._debug_live_count() == 0

    def test_negative_allocation_size_is_rejected(self) -> None:
        mr = NumpyHandlerMemoryResource()
        with pytest.raises(ValueError, match="-1"):
            mr.allocate(-1, _STREAM)

    def test_allocation_failure_raises_memory_error_with_context(self) -> None:
        mr = NumpyHandlerMemoryResource()
        # 2**61 bytes exceeds any 64-bit machine's address space, so the
        # captured handler's malloc must fail deterministically.
        with pytest.raises(MemoryError, match="cpu:0"):
            mr.allocate(1 << 61, _STREAM)
        assert mr._debug_live_count() == 0


class TestContracts:
    def test_capability_probes(self) -> None:
        mr = NumpyHandlerMemoryResource()
        # The captured handler could be anything the process configured, so
        # neither stream ordering nor alignment can be promised (design §5.1).
        assert mr.stream_ordered is False
        assert mr.guaranteed_alignment() == 1
        assert mr.available_memory() is None

    def test_non_cpu_devices_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="cpu"):
            NumpyHandlerMemoryResource(_CUDA)

    def test_repr_names_the_captured_handler(self) -> None:
        mr = NumpyHandlerMemoryResource()
        assert "default_allocator" in repr(mr)


class TestHandlerCapture:
    def test_allocations_go_through_the_currently_installed_handler(self) -> None:
        # The differential oracle: with a devmm handler installed (provide
        # direction), the consume MR must allocate through it — proving both
        # arrows meet at NumPy's real NEP-49 machinery.
        stats = StatisticsAdaptor(MallocMemoryResource())
        with integrations_numpy.install(stats):
            mr = NumpyHandlerMemoryResource()
            ptr = mr.allocate(512, _STREAM)
            assert stats.current_bytes == 512
            mr.deallocate(ptr, 512, _STREAM)
            assert stats.current_bytes == 0

    def test_the_handler_is_captured_at_construction_not_per_call(self) -> None:
        mr = NumpyHandlerMemoryResource()
        stats = StatisticsAdaptor(MallocMemoryResource())
        with integrations_numpy.install(stats):
            ptr = mr.allocate(256, _STREAM)
            assert stats.current_bytes == 0
            mr.deallocate(ptr, 256, _STREAM)


class TestRangeGuard:
    @pytest.mark.parametrize("version", ["1.21.6", "3.0.0"])
    def test_out_of_range_numpy_raises_cleanly(
        self, monkeypatch: pytest.MonkeyPatch, version: str
    ) -> None:
        monkeypatch.setattr(np, "__version__", version)
        with pytest.raises(RuntimeError, match=version.replace(".", r"\.")):
            NumpyHandlerMemoryResource()

    def test_unparseable_numpy_version_raises(self) -> None:
        with pytest.raises(RuntimeError, match="garbage"):
            _nep49.parsed_version("garbage")

    def test_missing_numpy_raises_runtime_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A None entry makes `import numpy` fail without uninstalling it.
        monkeypatch.setitem(sys.modules, "numpy", None)
        with pytest.raises(RuntimeUnavailableError, match="numpy"):
            _nep49._numpy_module()

    def test_missing_multiarray_umath_extension_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "numpy._core._multiarray_umath", None)
        monkeypatch.setitem(sys.modules, "numpy.core._multiarray_umath", None)
        with pytest.raises(RuntimeError, match="_multiarray_umath"):
            _nep49._multiarray_umath(np)
