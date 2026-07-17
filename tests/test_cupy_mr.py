"""`CupyAllocatorMemoryResource` over a scripted CuPy double (design §5.2,
§9): the allocator call runs with the MR's device and the caller's stream
current (CuPy pools key cached blocks by the thread-local current stream),
the returned `MemoryPointer` is stashed until `deallocate` drops it, and the
sibling-MR misuse contracts hold.
"""

from __future__ import annotations

import gc
import importlib.util
import weakref
from typing import Any

import pytest

from devmm import Device
from devmm._runtimes.base import RuntimeUnavailableError
from devmm.integrations._support import ForeignHandleStream
from devmm.mrs import cuda as mrs_cuda
from devmm.mrs.cuda import CupyAllocatorMemoryResource
from tests._integration_fakes import FakeCupy, FakeCupyMemory, FakeCupyMemoryPointer

_DEVICE = Device.from_string("cuda:1")
_CPU = Device.from_string("cpu")


@pytest.fixture
def fake_cupy(monkeypatch: pytest.MonkeyPatch) -> FakeCupy:
    fake = FakeCupy()
    monkeypatch.setattr(mrs_cuda, "_cupy_module", lambda: fake)
    return fake


def _stream(handle: int = 0x7000) -> ForeignHandleStream:
    return ForeignHandleStream(_DEVICE, handle)


class TestAllocation:
    def test_allocate_runs_inside_device_and_stream_contexts(self, fake_cupy: FakeCupy) -> None:
        mr = CupyAllocatorMemoryResource(device=_DEVICE)
        mr.allocate(64, _stream())
        assert fake_cupy.events == [
            ("device_enter", 1),
            ("stream_enter", 0x7000),
            ("alloc", 64, 1, 0x7000),
            ("stream_exit", 0x7000),
            ("device_exit", 1),
        ]

    def test_default_allocator_is_cupy_alloc(self, fake_cupy: FakeCupy) -> None:
        mr = CupyAllocatorMemoryResource(device=_DEVICE)
        ptr = mr.allocate(32, _stream())
        assert ("alloc", 32, 1, 0x7000) in fake_cupy.events
        assert ptr != 0

    def test_an_explicit_allocator_wins(self, fake_cupy: FakeCupy) -> None:
        calls: list[int] = []

        def pool_malloc(nbytes: int) -> FakeCupyMemoryPointer:
            calls.append(nbytes)
            return FakeCupyMemoryPointer(FakeCupyMemory(0x9000, nbytes), 0)

        mr = CupyAllocatorMemoryResource(pool_malloc, _DEVICE)
        ptr = mr.allocate(128, _stream())
        assert calls == [128]
        assert ptr == 0x9000
        assert all(event[0] != "alloc" for event in fake_cupy.events)

    def test_returned_pointer_is_the_memory_pointers(self, fake_cupy: FakeCupy) -> None:
        mr = CupyAllocatorMemoryResource(device=_DEVICE)
        stream = _stream()
        ptr = mr.allocate(64, stream)
        assert mr._debug_live_count() == 1
        mr.deallocate(ptr, 64, stream)
        assert mr._debug_live_count() == 0

    def test_deallocate_drops_the_memory_pointer(self, fake_cupy: FakeCupy) -> None:
        seen: list[FakeCupyMemoryPointer] = []

        def allocator(nbytes: int) -> FakeCupyMemoryPointer:
            memptr = FakeCupyMemoryPointer(FakeCupyMemory(0xA000, nbytes), 0)
            seen.append(memptr)
            return memptr

        mr = CupyAllocatorMemoryResource(allocator, _DEVICE)
        stream = _stream()
        ptr = mr.allocate(64, stream)
        ref = weakref.ref(seen.pop())
        gc.collect()
        # The MR's stash is what keeps CuPy's block alive (design §5.2).
        assert ref() is not None
        mr.deallocate(ptr, 64, stream)
        gc.collect()
        assert ref() is None

    def test_zero_byte_requests_are_bumped_to_one_byte(self, fake_cupy: FakeCupy) -> None:
        mr = CupyAllocatorMemoryResource(device=_DEVICE)
        stream = _stream()
        first = mr.allocate(0, stream)
        second = mr.allocate(0, stream)
        assert first != 0
        assert second != 0
        assert first != second
        assert ("alloc", 1, 1, 0x7000) in fake_cupy.events
        mr.deallocate(first, 0, stream)
        mr.deallocate(second, 0, stream)
        assert mr._debug_live_count() == 0

    def test_negative_sizes_are_rejected_without_touching_cupy(self, fake_cupy: FakeCupy) -> None:
        mr = CupyAllocatorMemoryResource(device=_DEVICE)
        with pytest.raises(ValueError, match="-1"):
            mr.allocate(-1, _stream())
        assert fake_cupy.events == []

    def test_allocator_memory_errors_propagate_untouched(self, fake_cupy: FakeCupy) -> None:
        # CuPy's OutOfMemoryError subclasses MemoryError, so the MR forwards
        # allocator failures as-is (design §3.3, §5.2).
        marker = MemoryError("pool exhausted")

        def exploding(nbytes: int) -> Any:
            raise marker

        mr = CupyAllocatorMemoryResource(exploding, _DEVICE)
        with pytest.raises(MemoryError) as excinfo:
            mr.allocate(64, _stream())
        assert excinfo.value is marker
        assert mr._debug_live_count() == 0


class TestMisuse:
    def test_double_free_raises(self, fake_cupy: FakeCupy) -> None:
        mr = CupyAllocatorMemoryResource(device=_DEVICE)
        stream = _stream()
        ptr = mr.allocate(64, stream)
        mr.deallocate(ptr, 64, stream)
        with pytest.raises(ValueError, match="live allocation"):
            mr.deallocate(ptr, 64, stream)

    def test_foreign_pointer_free_raises(self, fake_cupy: FakeCupy) -> None:
        mr = CupyAllocatorMemoryResource(device=_DEVICE)
        stream = _stream()
        ptr = mr.allocate(64, stream)
        with pytest.raises(ValueError, match="live allocation"):
            mr.deallocate(ptr + 1, 64, stream)
        mr.deallocate(ptr, 64, stream)

    def test_size_mismatched_free_raises_and_keeps_the_allocation_live(
        self, fake_cupy: FakeCupy
    ) -> None:
        mr = CupyAllocatorMemoryResource(device=_DEVICE)
        stream = _stream()
        ptr = mr.allocate(64, stream)
        with pytest.raises(ValueError, match="size mismatch"):
            mr.deallocate(ptr, 63, stream)
        assert mr._debug_live_count() == 1
        mr.deallocate(ptr, 64, stream)


class TestContracts:
    def test_capability_probes(self) -> None:
        mr = CupyAllocatorMemoryResource(device=_DEVICE)
        assert mr.stream_ordered is True
        assert mr.guaranteed_alignment() == 256
        assert mr.device == _DEVICE

    def test_non_cuda_devices_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="cuda"):
            CupyAllocatorMemoryResource(device=_CPU)
        with pytest.raises(ValueError, match="cuda"):
            CupyAllocatorMemoryResource(device=Device.from_string("rocm:0"))

    def test_repr_names_the_device(self) -> None:
        mr = CupyAllocatorMemoryResource(device=_DEVICE)
        assert "cuda:1" in repr(mr)
        assert type(mr).__name__ in repr(mr)


@pytest.mark.skipif(importlib.util.find_spec("cupy") is not None, reason="cupy is installed here")
class TestWithoutCupy:
    def test_construction_is_lazy(self) -> None:
        # cupy is imported per allocation, so building (and never using) the
        # MR works on machines without it.
        CupyAllocatorMemoryResource(device=_DEVICE)

    def test_allocate_raises_runtime_unavailable(self) -> None:
        mr = CupyAllocatorMemoryResource(device=_DEVICE)
        with pytest.raises(RuntimeUnavailableError, match=r"devmm\[cupy\]"):
            mr.allocate(64, _stream())
