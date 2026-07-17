"""`integrations.numba` over a scripted Numba double (design §5.4, §6, §9):
the lazily built `DevmmEMMPlugin` allocates through the current devmm MR
(registry-resolved per allocation), its finalizer frees exactly once and
tolerates Numba's context-reset teardown, and `install()` drives
`set_memory_manager` reversibly — restore on uninstall, exit and exception.
"""

from __future__ import annotations

import ctypes
import gc
import importlib.util
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from devmm import Device, using_memory_resource
from devmm._core.buffer import DeviceBuffer
from devmm._runtimes.base import RuntimeUnavailableError
from devmm.integrations import numba as integrations_numba
from devmm.testing import RecordingMemoryResource
from tests._integration_fakes import (
    FakeNumbaCuda,
    FakeNumbaMemoryInfo,
    FakeNumbaMemoryPointer,
)

_DEVICE = Device.from_string("cuda:0")


@pytest.fixture
def fake_numba(monkeypatch: pytest.MonkeyPatch) -> FakeNumbaCuda:
    fake = FakeNumbaCuda()
    package = ModuleType("numba")
    package.cuda = fake  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "numba", package)
    monkeypatch.setitem(sys.modules, "numba.cuda", fake)
    monkeypatch.setattr(integrations_numba, "_plugin_class_cache", None)
    return fake


def _manager(device_index: int = 0) -> Any:
    context = SimpleNamespace(device=SimpleNamespace(id=device_index))
    return integrations_numba._plugin_class()(context=context)


class TestLazyExports:
    def test_devmm_emm_plugin_and_the_numba_hook_name_are_the_same_class(
        self, fake_numba: FakeNumbaCuda
    ) -> None:
        plugin = integrations_numba.DevmmEMMPlugin
        # `_numba_memory_manager` is the name Numba's
        # NUMBA_CUDA_MEMORY_MANAGER env hook reads off the module.
        assert integrations_numba._numba_memory_manager is plugin
        assert integrations_numba.DevmmEMMPlugin is plugin

    def test_unknown_attributes_still_raise(self, fake_numba: FakeNumbaCuda) -> None:
        with pytest.raises(AttributeError, match="nope"):
            _ = integrations_numba.nope

    @pytest.mark.skipif(
        importlib.util.find_spec("numba") is not None, reason="numba is installed here"
    )
    def test_without_numba_the_lazy_class_raises_runtime_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(integrations_numba, "_plugin_class_cache", None)
        with pytest.raises(RuntimeUnavailableError, match=r"devmm\[numba\]"):
            _ = integrations_numba.DevmmEMMPlugin


class TestPlugin:
    def test_interface_version_is_1(self, fake_numba: FakeNumbaCuda) -> None:
        plugin = integrations_numba._plugin_class()(context=None)
        assert plugin.interface_version == 1

    def test_memalloc_allocates_through_the_current_devmm_mr(
        self, fake_numba: FakeNumbaCuda
    ) -> None:
        mr = RecordingMemoryResource(_DEVICE)
        manager = _manager()
        with using_memory_resource(mr):
            memory = manager.memalloc(256)
        assert isinstance(memory, FakeNumbaMemoryPointer)
        name, ptr, nbytes, stream = mr.calls[0]
        assert (name, nbytes) == ("allocate", 256)
        assert isinstance(memory.pointer, ctypes.c_uint64)
        assert memory.pointer.value == ptr
        assert memory.size == 256
        # The EMM protocol carries no stream, so allocations ride the
        # default stream — the same choice rmm's plugin makes.
        assert stream.handle == 0
        assert stream.device == _DEVICE
        buffer = manager.allocations[ptr]
        assert isinstance(buffer, DeviceBuffer)
        assert buffer.ptr == ptr

    def test_the_finalizer_frees_exactly_once(self, fake_numba: FakeNumbaCuda) -> None:
        mr = RecordingMemoryResource(_DEVICE)
        manager = _manager()
        with using_memory_resource(mr):
            memory = manager.memalloc(64)
        memory.free()
        deallocations = [call for call in mr.calls if call[0] == "deallocate"]
        assert len(deallocations) == 1
        assert manager.allocations == {}
        assert mr.live == {}

    def test_the_finalizer_tolerates_a_context_reset(self, fake_numba: FakeNumbaCuda) -> None:
        # At teardown Numba may clear the allocations mapping before device
        # arrays die; the buffer's own safety net then releases the memory
        # and the late finalizer must find nothing to free — one
        # deallocation total, no double-free, no KeyError.
        mr = RecordingMemoryResource(_DEVICE)
        manager = _manager()
        with using_memory_resource(mr):
            memory = manager.memalloc(64)
        manager.allocations.clear()
        gc.collect()
        assert len([call for call in mr.calls if call[0] == "deallocate"]) == 1
        memory.free()
        assert len([call for call in mr.calls if call[0] == "deallocate"]) == 1
        assert mr.live == {}

    def test_get_memory_info_reports_the_mrs_available_memory(
        self, fake_numba: FakeNumbaCuda
    ) -> None:
        class _InformedMr(RecordingMemoryResource):
            def available_memory(self) -> tuple[int, int] | None:
                return (123, 456)

        manager = _manager()
        with using_memory_resource(_InformedMr(_DEVICE)):
            info = manager.get_memory_info()
        assert isinstance(info, FakeNumbaMemoryInfo)
        assert (info.free, info.total) == (123, 456)

    def test_get_memory_info_without_mr_support_raises(self, fake_numba: FakeNumbaCuda) -> None:
        manager = _manager()
        with (
            using_memory_resource(RecordingMemoryResource(_DEVICE)),
            pytest.raises(NotImplementedError, match="available memory"),
        ):
            manager.get_memory_info()


class TestInstall:
    def test_install_sets_the_manager_and_uninstall_restores_none(
        self, fake_numba: FakeNumbaCuda
    ) -> None:
        driver = fake_numba.cudadrv.driver
        assert driver._memory_manager is None
        handle = integrations_numba.install()
        assert driver._memory_manager is integrations_numba._plugin_class()
        # set_memory_manager cannot express "back to the built-in manager"
        # (it instantiates its argument), so restore goes through the global.
        handle.uninstall()
        assert driver._memory_manager is None

    def test_install_restores_a_previous_manager(self, fake_numba: FakeNumbaCuda) -> None:
        previous = type("PreviousManager", (), {})
        driver = fake_numba.cudadrv.driver
        driver._memory_manager = previous
        with integrations_numba.install():
            assert driver._memory_manager is integrations_numba._plugin_class()
        assert driver._memory_manager is previous

    def test_context_manager_restores_on_exception(self, fake_numba: FakeNumbaCuda) -> None:
        driver = fake_numba.cudadrv.driver
        with pytest.raises(RuntimeError, match="boom"):  # noqa: SIM117 - the point is nesting
            with integrations_numba.install():
                raise RuntimeError("boom")
        assert driver._memory_manager is None

    def test_install_goes_through_numbas_validating_setter(self, fake_numba: FakeNumbaCuda) -> None:
        integrations_numba.install()
        assert fake_numba.memory_manager_calls == [integrations_numba._plugin_class()]
