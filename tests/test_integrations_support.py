"""The shared provide-direction plumbing (design §6): the `Installer`
contract every `install()` returns (restore exactly once, on exit or
exception, never clobbering a newer installation), the direct
consume+provide cycle guard, and the foreign-stream shims that carry
third-party stream handles into MR calls.
"""

from __future__ import annotations

from typing import Any

import pytest

from devmm import Device, LoggingAdaptor, StatisticsAdaptor
from devmm._core.stream import Stream
from devmm.integrations import Installer, _support
from devmm.integrations._support import ForeignHandleStream, ensure_no_cycle, stream_handle_of
from devmm.testing import RecordingMemoryResource
from tests._integration_fakes import FakeForeignStream

_CUDA0 = Device.from_string("cuda:0")


class TestInstaller:
    def test_uninstall_runs_restore_exactly_once(self) -> None:
        calls: list[str] = []
        handle = Installer("Lib", lambda: calls.append("restore"))
        assert handle.installed
        handle.uninstall()
        assert calls == ["restore"]
        assert not handle.installed
        handle.uninstall()
        assert calls == ["restore"]

    def test_context_manager_restores_on_normal_exit(self) -> None:
        calls: list[str] = []
        with Installer("Lib", lambda: calls.append("restore")) as handle:
            assert handle.installed
            assert calls == []
        assert calls == ["restore"]
        assert not handle.installed

    def test_context_manager_restores_on_exception(self) -> None:
        calls: list[str] = []
        with pytest.raises(RuntimeError, match="boom"):  # noqa: SIM117 - the point is nesting
            with Installer("Lib", lambda: calls.append("restore")):
                raise RuntimeError("boom")
        assert calls == ["restore"]

    def test_spent_handle_does_not_clobber_a_newer_installation(self) -> None:
        state = {"hook": "original"}
        handle = Installer("Lib", lambda: state.__setitem__("hook", "original"))
        handle.uninstall()
        state["hook"] = "newer"
        handle.uninstall()
        assert state["hook"] == "newer"

    def test_repr_names_the_library_and_state(self) -> None:
        handle = Installer("CuPy", lambda: None)
        assert "CuPy" in repr(handle)
        assert "installed" in repr(handle)
        handle.uninstall()
        assert "uninstalled" in repr(handle)


class _ForbiddenMr(RecordingMemoryResource):
    """Stands in for a consume-direction MR of some library."""


class TestEnsureNoCycle:
    def test_direct_forbidden_mr_raises(self) -> None:
        mr = _ForbiddenMr()
        with pytest.raises(ValueError, match="cycle") as excinfo:
            ensure_no_cycle(mr, _ForbiddenMr, "SomeLib")
        assert "SomeLib" in str(excinfo.value)
        assert "_ForbiddenMr" in str(excinfo.value)

    def test_forbidden_mr_anywhere_in_the_adaptor_chain_raises(self) -> None:
        mr = StatisticsAdaptor(LoggingAdaptor(_ForbiddenMr()))
        with pytest.raises(ValueError, match="cycle"):
            ensure_no_cycle(mr, _ForbiddenMr, "SomeLib")

    def test_clean_chain_passes(self) -> None:
        mr = StatisticsAdaptor(LoggingAdaptor(RecordingMemoryResource()))
        ensure_no_cycle(mr, _ForbiddenMr, "SomeLib")


class _StubStream(Stream):
    def __init__(self, device: Device) -> None:
        self.device = device
        self.calls: list[tuple[Any, ...]] = []

    @property
    def handle(self) -> int:
        return 0

    def synchronize(self) -> None:
        self.calls.append(("synchronize",))

    def wait_raw(self, other_handle: int) -> None:
        self.calls.append(("wait_raw", other_handle))


class _StubRuntime:
    def __init__(self, stream: _StubStream) -> None:
        self.stream = stream
        self.calls: list[tuple[Any, ...]] = []

    def wrap_stream(self, device: Device, obj: object) -> Stream:
        self.calls.append(("wrap_stream", device, obj))
        return self.stream


class TestForeignHandleStream:
    def test_carries_device_and_handle(self) -> None:
        stream = ForeignHandleStream(_CUDA0, 0x7000)
        assert stream.device == _CUDA0
        assert stream.handle == 0x7000
        assert stream.__cuda_stream__() == (0, 0x7000)

    def test_negative_handles_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="-1"):
            ForeignHandleStream(_CUDA0, -1)

    def test_ordering_primitives_resolve_the_device_runtime(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        native = _StubStream(_CUDA0)
        runtime = _StubRuntime(native)
        monkeypatch.setattr(_support, "runtime_for", lambda device: runtime)
        stream = ForeignHandleStream(_CUDA0, 0xAB)
        stream.synchronize()
        stream.wait_raw(0xCD)
        assert runtime.calls == [("wrap_stream", _CUDA0, 0xAB), ("wrap_stream", _CUDA0, 0xAB)]
        assert native.calls == [("synchronize",), ("wait_raw", 0xCD)]


class TestStreamHandleOf:
    def test_none_means_the_default_stream(self) -> None:
        assert stream_handle_of(None) == 0

    def test_ints_pass_through(self) -> None:
        assert stream_handle_of(0x123) == 0x123

    def test_cuda_stream_protocol_objects_yield_their_handle(self) -> None:
        assert stream_handle_of(FakeForeignStream(0x456)) == 0x456

    def test_non_int_protocol_handles_are_rejected(self) -> None:
        class _Bad:
            def __cuda_stream__(self) -> tuple[int, object]:
                return (0, "nope")

        with pytest.raises(TypeError, match="__cuda_stream__"):
            stream_handle_of(_Bad())

    @pytest.mark.parametrize("attribute", ["handle", "ptr", "value"])
    def test_ecosystem_attribute_spellings_are_read(self, attribute: str) -> None:
        class _Foreign:
            pass

        obj = _Foreign()
        setattr(obj, attribute, 0x789)
        assert stream_handle_of(obj) == 0x789

    def test_unreadable_objects_are_rejected(self) -> None:
        with pytest.raises(TypeError, match="stream handle"):
            stream_handle_of(object())
