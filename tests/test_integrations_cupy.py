"""`integrations.cupy.install` over a scripted CuPy double (design §6, §9):
the allocator hook registry round trip (install -> set_allocator, uninstall
-> previous allocator restored, exception path included), the direct
consume+provide cycle guard, and the installed hook's allocation behavior —
every CuPy allocation becomes a devmm `DeviceBuffer` released back through
the MR when the consumer drops it.
"""

from __future__ import annotations

import gc
import sys
import weakref

import pytest

from devmm import Device, StatisticsAdaptor
from devmm._core.buffer import DeviceBuffer
from devmm._runtimes.base import RuntimeUnavailableError
from devmm.integrations import cupy as integrations_cupy
from devmm.mrs.cuda import CupyAllocatorMemoryResource
from devmm.testing import RecordingMemoryResource
from tests._integration_fakes import (
    FakeCupy,
    FakeCupyMemoryPointer,
    FakeCupyStream,
    FakeCupyUnownedMemory,
)

_DEVICE = Device.from_string("cuda:0")


@pytest.fixture
def fake_cupy(monkeypatch: pytest.MonkeyPatch) -> FakeCupy:
    fake = FakeCupy()
    monkeypatch.setattr(integrations_cupy, "_cupy_module", lambda: fake)
    return fake


def _mr() -> RecordingMemoryResource:
    return RecordingMemoryResource(_DEVICE)


class TestInstallRoundTrip:
    def test_install_sets_and_uninstall_restores_the_allocator(self, fake_cupy: FakeCupy) -> None:
        previous = fake_cupy.cuda.get_allocator()
        handle = integrations_cupy.install(_mr())
        assert fake_cupy.cuda.get_allocator() is not previous
        handle.uninstall()
        assert fake_cupy.cuda.get_allocator() is previous

    def test_context_manager_restores_on_exception(self, fake_cupy: FakeCupy) -> None:
        previous = fake_cupy.cuda.get_allocator()
        with pytest.raises(RuntimeError, match="boom"):  # noqa: SIM117 - the point is nesting
            with integrations_cupy.install(_mr()):
                assert fake_cupy.cuda.get_allocator() is not previous
                raise RuntimeError("boom")
        assert fake_cupy.cuda.get_allocator() is previous

    def test_a_spent_handle_does_not_clobber_a_newer_allocator(self, fake_cupy: FakeCupy) -> None:
        handle = integrations_cupy.install(_mr())
        handle.uninstall()

        def newer(nbytes: int) -> FakeCupyMemoryPointer:
            raise AssertionError("never called")

        fake_cupy.cuda.set_allocator(newer)
        handle.uninstall()
        assert fake_cupy.cuda.get_allocator() is newer


class TestInstalledHook:
    def test_the_hook_allocates_from_the_mr_on_cupys_current_stream(
        self, fake_cupy: FakeCupy
    ) -> None:
        mr = _mr()
        with integrations_cupy.install(mr):
            fake_cupy.cuda.current_stream = FakeCupyStream(0x77)
            memory_pointer = fake_cupy.cuda.get_allocator()(512)
        assert isinstance(memory_pointer, FakeCupyMemoryPointer)
        name, ptr, nbytes, stream = mr.calls[0]
        assert (name, nbytes) == ("allocate", 512)
        assert memory_pointer.ptr == ptr
        assert stream.handle == 0x77
        assert stream.device == _DEVICE

    def test_the_memory_pointer_owns_a_device_buffer_over_the_mr(self, fake_cupy: FakeCupy) -> None:
        mr = _mr()
        with integrations_cupy.install(mr):
            memory_pointer = fake_cupy.cuda.get_allocator()(512)
        memory = memory_pointer.mem
        assert isinstance(memory, FakeCupyUnownedMemory)
        assert isinstance(memory.owner, DeviceBuffer)
        assert memory.size == 512
        assert memory.ptr == memory.owner.ptr
        # A live pointer lets CuPy infer the device from it (the rmm
        # allocator makes the same -1 distinction).
        assert memory.device_id == -1

    def test_dropping_the_consumer_releases_the_buffer_through_the_mr(
        self, fake_cupy: FakeCupy
    ) -> None:
        mr = _mr()
        with integrations_cupy.install(mr):
            memory_pointer = fake_cupy.cuda.get_allocator()(512)
        buffer_ref = weakref.ref(memory_pointer.mem.owner)
        del memory_pointer
        gc.collect()
        assert buffer_ref() is None
        deallocations = [call for call in mr.calls if call[0] == "deallocate"]
        assert len(deallocations) == 1
        assert mr.live == {}

    def test_allocations_survive_uninstall_until_the_consumer_drops_them(
        self, fake_cupy: FakeCupy
    ) -> None:
        mr = _mr()
        handle = integrations_cupy.install(mr)
        memory_pointer = fake_cupy.cuda.get_allocator()(256)
        handle.uninstall()
        assert len(mr.live) == 1
        del memory_pointer
        gc.collect()
        assert mr.live == {}


class TestRefusals:
    def test_composing_the_consume_and_provide_arrows_raises(self) -> None:
        # No fake is wired: the cycle guard must fire before cupy is even
        # imported (ValueError, not RuntimeUnavailableError).
        mr = CupyAllocatorMemoryResource(device=_DEVICE)
        with pytest.raises(ValueError, match="cycle"):
            integrations_cupy.install(mr)

    def test_the_cycle_is_detected_through_the_adaptor_chain(self) -> None:
        mr = StatisticsAdaptor(CupyAllocatorMemoryResource(device=_DEVICE))
        with pytest.raises(ValueError, match="cycle"):
            integrations_cupy.install(mr)

    def test_non_cuda_mrs_are_rejected_before_importing_cupy(self) -> None:
        with pytest.raises(ValueError, match="cuda"):
            integrations_cupy.install(RecordingMemoryResource())


class TestCupyModuleSeam:
    def test_returns_the_importable_cupy_module(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sentinel = FakeCupy()
        monkeypatch.setitem(sys.modules, "cupy", sentinel)
        assert integrations_cupy._cupy_module() is sentinel

    def test_missing_cupy_raises_runtime_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A None entry makes `import cupy` fail without uninstalling it.
        monkeypatch.setitem(sys.modules, "cupy", None)
        with pytest.raises(RuntimeUnavailableError, match=r"devmm\[cupy\]"):
            integrations_cupy._cupy_module()
