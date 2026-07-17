"""The CUDA and ROCm `DeviceRuntime`s over a scripted `FakeGpuApi` (design
§4, §9), one shared suite parametrized over both platform harnesses: call
sequences and failure injection for streams, memcpy, device activation and
the DLPack handoff primitive; the driver-keyed discovery wiring with the
`DEVMM_RUNTIME` override; the `rmm`-module platform disambiguation (§4.2);
and the runtime-backed default stream/default MR for `empty()` — no hardware.
"""

from __future__ import annotations

import ctypes
import gc
import sys
import types
from collections.abc import Iterator
from typing import cast

import pytest

from devmm import (
    Device,
    DeviceType,
    available_runtimes,
    empty,
    get_current_memory_resource,
    runtime_for,
    runtime_names,
)
from devmm._core import registry as registry_module
from devmm._core.stream import CpuStream, StreamError
from devmm._runtimes import _discovery, _gpulib
from devmm._runtimes.base import CopyKind, DeviceRuntime, RuntimeUnavailableError
from devmm._runtimes.cuda import CUDA_PLATFORM, CudaRuntime
from devmm._runtimes.rocm import HIP_PLATFORM, HipRuntime
from devmm.testing import RecordingMemoryResource
from tests._gpu_fakes import HARNESSES, AsynclessNativeLib, FakeGpuApi, FakeRmmMemoryResource
from tests._gpu_fakes import GpuHarness as Harness

_CPU = Device.from_string("cpu")

# Structural SPI conformance, verified by mypy --strict: both platform
# runtimes satisfy the `DeviceRuntime` protocol.
_SPI_CHECKS: tuple[DeviceRuntime, ...] = (
    CudaRuntime(api=FakeGpuApi(CUDA_PLATFORM)),
    HipRuntime(api=FakeGpuApi(HIP_PLATFORM)),
)


@pytest.fixture(params=sorted(HARNESSES))
def h(request: pytest.FixtureRequest) -> Harness:
    return HARNESSES[cast(str, request.param)]


def _runtime(h: Harness, **api_kwargs: int | bool) -> tuple[_gpulib.GpuRuntime, FakeGpuApi]:
    api = h.api(**api_kwargs)
    return h.runtime_cls(api=api), api


@pytest.fixture(autouse=True)
def _isolated_discovery(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Fresh runtime/spec caches and no ambient `DEVMM_RUNTIME` per test."""
    monkeypatch.delenv("DEVMM_RUNTIME", raising=False)
    saved = dict(_discovery._loaded)
    _discovery._loaded.clear()
    _discovery._clear_spec_cache()
    yield
    _discovery._loaded.clear()
    _discovery._loaded.update(saved)
    _discovery._clear_spec_cache()


@pytest.fixture
def _isolated_registry() -> Iterator[None]:
    """Snapshot and restore the process-wide current-MR registry."""
    saved = dict(registry_module._registry)
    registry_module._registry.clear()
    yield
    registry_module._registry.clear()
    registry_module._registry.update(saved)


def _install_fake_runtime(
    monkeypatch: pytest.MonkeyPatch, h: Harness, api: FakeGpuApi
) -> _gpulib.GpuRuntime:
    """Replace the harness's built-in spec with one resolving to a fake-api
    runtime."""
    runtime = h.runtime_cls(api=api)
    specs = tuple(
        _discovery._RuntimeSpec(h.name, lambda: True, lambda: runtime)
        if spec.name == h.name
        else spec
        for spec in _discovery._BUILTIN_SPECS
    )
    monkeypatch.setattr(_discovery, "_BUILTIN_SPECS", specs)
    _discovery._clear_spec_cache()
    return runtime


def _rmm_module(*marker_classes: str) -> types.ModuleType:
    """An `rmm`-named module double whose `mr` namespace carries the given
    platform marker classes (design §4.2)."""
    module = types.ModuleType("rmm")
    mr_namespace = types.SimpleNamespace()
    for name in marker_classes:
        setattr(mr_namespace, name, type(name, (), {}))
    module.mr = mr_namespace  # type: ignore[attr-defined]
    return module


class TestIdentity:
    def test_identity(self, h: Harness) -> None:
        runtime, _ = _runtime(h)
        assert runtime.name == h.name
        assert runtime.device_types == frozenset({h.platform.device_type})

    def test_device_count_queries_the_api(self, h: Harness) -> None:
        runtime, api = _runtime(h, device_count=3)
        assert runtime.device_count(h.platform.device_type) == 3
        assert api.calls == [(h.platform.symbols.get_device_count,)]

    def test_device_count_is_zero_for_foreign_device_types(self, h: Harness) -> None:
        runtime, api = _runtime(h)
        assert runtime.device_count(DeviceType.CPU) == 0
        assert runtime.device_count(h.foreign_device_type) == 0
        assert api.calls == []

    def test_device_count_failure_raises_the_platform_error(self, h: Harness) -> None:
        runtime, api = _runtime(h)
        api.fail[h.platform.symbols.get_device_count] = 999
        with pytest.raises(h.error_cls, match=f"fake {h.name} error 999"):
            runtime.device_count(h.platform.device_type)

    def test_unloadable_runtime_library_raises_runtime_unavailable(
        self, h: Harness, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(_gpulib, "load_first_library", lambda names: None)
        with pytest.raises(RuntimeUnavailableError, match=h.library_match):
            h.runtime_cls()


class TestNativeApi:
    def test_missing_async_symbols_report_not_supported(self, h: Harness) -> None:
        # A runtime library predating the async allocation family lacks the
        # malloc_async/free_async pair; the api shim constructs anyway and
        # the async family reports the platform's not-supported status
        # instead of raising AttributeError.
        api = _gpulib.NativeGpuApi(cast(ctypes.CDLL, AsynclessNativeLib(h.platform)), h.platform)
        assert api.malloc_async(64, 0) == (h.platform.not_supported_status, 0)
        assert api.free_async(0x1000, 0) == h.platform.not_supported_status


class TestStreams:
    def test_default_stream_is_the_default_handle(self, h: Harness) -> None:
        runtime, api = _runtime(h)
        stream = runtime.default_stream(h.device1)
        assert isinstance(stream, h.stream_cls)
        assert stream.handle == 0
        assert stream.device == h.device1
        assert api.calls == []

    def test_create_stream_activates_the_device_and_creates(self, h: Harness) -> None:
        runtime, api = _runtime(h, current_device=0)
        stream = runtime.create_stream(h.device1)
        symbols = h.platform.symbols
        assert api.calls == [
            (symbols.get_device,),
            (symbols.set_device, 1),
            (symbols.stream_create,),
            (symbols.set_device, 0),
        ]
        assert stream.device == h.device1
        assert stream.handle in api.live_streams

    def test_created_stream_is_destroyed_with_its_wrapper(self, h: Harness) -> None:
        runtime, api = _runtime(h)
        stream = runtime.create_stream(h.device0)
        handle = stream.handle
        del stream
        gc.collect()
        assert (h.platform.symbols.stream_destroy, handle) in api.calls
        assert handle not in api.live_streams

    def test_create_stream_failure_raises_the_platform_error(self, h: Harness) -> None:
        runtime, api = _runtime(h)
        api.fail[h.platform.symbols.stream_create] = 999
        with pytest.raises(h.error_cls, match=h.platform.symbols.stream_create):
            runtime.create_stream(h.device0)

    def test_synchronize_routes_through_the_api(self, h: Harness) -> None:
        api = h.api()
        stream = h.stream_cls(h.device0, 7, api)
        stream.synchronize()
        assert api.calls == [(h.platform.symbols.stream_synchronize, 7)]

    def test_synchronize_failure_raises_the_platform_error(self, h: Harness) -> None:
        api = h.api()
        api.fail[h.platform.symbols.stream_synchronize] = 999
        with pytest.raises(h.error_cls, match=h.platform.symbols.stream_synchronize):
            h.stream_cls(h.device0, 7, api).synchronize()

    def test_wait_raw_orders_via_an_event(self, h: Harness) -> None:
        api = h.api()
        stream = h.stream_cls(h.device0, 7, api)
        stream.wait_raw(9)
        symbols = h.platform.symbols
        event = api.calls[1][1]
        assert api.calls == [
            (symbols.event_create_with_flags, h.platform.event_disable_timing),
            (symbols.event_record, event, 9),
            (symbols.stream_wait_event, 7, event, 0),
            (symbols.event_destroy, event),
        ]
        assert api.live_events == set()

    def test_stream_implements_the_cuda_stream_protocol(self, h: Harness) -> None:
        # `__cuda_stream__` is the cross-library stream protocol on both
        # platforms (CuPy-ROCm and hipMM speak it too, design §3.2).
        assert h.stream_cls(h.device0, 0xBEEF, h.api()).__cuda_stream__() == (0, 0xBEEF)

    def test_stream_rejects_cpu_devices(self, h: Harness) -> None:
        with pytest.raises(ValueError, match="cpu"):
            h.stream_cls(_CPU, 0, h.api())

    def test_stream_rejects_the_sibling_platform_devices(self, h: Harness) -> None:
        foreign = Device(h.foreign_device_type, 0)
        with pytest.raises(ValueError, match=h.name):
            h.stream_cls(foreign, 0, h.api())

    def test_stream_rejects_negative_handles(self, h: Harness) -> None:
        with pytest.raises(ValueError, match="-1"):
            h.stream_cls(h.device0, -1, h.api())


class TestWrapStream:
    def test_passes_through_a_stream_on_the_same_device(self, h: Harness) -> None:
        runtime, api = _runtime(h)
        stream = h.stream_cls(h.device0, 5, api)
        assert runtime.wrap_stream(h.device0, stream) is stream

    def test_rejects_streams_on_other_devices(self, h: Harness) -> None:
        runtime, _ = _runtime(h)
        with pytest.raises(ValueError, match="cpu"):
            runtime.wrap_stream(h.device0, CpuStream())

    def test_maps_sentinels_to_the_platform_magic_handles(self, h: Harness) -> None:
        runtime, _ = _runtime(h)
        for sentinel, handle in h.expected_sentinels:
            stream = runtime.wrap_stream(h.device0, sentinel)
            assert isinstance(stream, h.stream_cls)
            assert stream.handle == handle

    def test_wraps_raw_int_handles(self, h: Harness) -> None:
        runtime, _ = _runtime(h)
        stream = runtime.wrap_stream(h.device1, 0xBEEF)
        assert isinstance(stream, h.stream_cls)
        assert stream.handle == 0xBEEF
        assert stream.device == h.device1

    def test_rejects_negative_handles(self, h: Harness) -> None:
        runtime, _ = _runtime(h)
        with pytest.raises(ValueError, match="-3"):
            runtime.wrap_stream(h.device0, -3)

    def test_accepts_cuda_stream_protocol_objects(self, h: Harness) -> None:
        class _Proto:
            def __cuda_stream__(self) -> tuple[int, int]:
                return (0, 0xF00D)

        runtime, _ = _runtime(h)
        stream = runtime.wrap_stream(h.device0, _Proto())
        assert isinstance(stream, h.stream_cls)
        assert stream.handle == 0xF00D

    def test_rejects_non_int_protocol_handles(self, h: Harness) -> None:
        class _BadProto:
            def __cuda_stream__(self) -> tuple[int, str]:
                return (0, "seven")

        runtime, _ = _runtime(h)
        with pytest.raises(TypeError):
            runtime.wrap_stream(h.device0, _BadProto())

    def test_rejects_unknown_objects(self, h: Harness) -> None:
        runtime, _ = _runtime(h)
        with pytest.raises(TypeError):
            runtime.wrap_stream(h.device0, object())

    def test_rejects_cpu_devices(self, h: Harness) -> None:
        runtime, _ = _runtime(h)
        with pytest.raises(ValueError, match="cpu"):
            runtime.wrap_stream(_CPU, 0)


class TestMakeStreamWait:
    def test_emits_create_record_wait_destroy(self, h: Harness) -> None:
        runtime, api = _runtime(h)
        producer = h.stream_cls(h.device0, 3, api)
        runtime.make_stream_wait(5, producer)
        symbols = h.platform.symbols
        event = api.calls[1][1]
        assert api.calls == [
            (symbols.event_create_with_flags, h.platform.event_disable_timing),
            (symbols.event_record, event, 3),
            (symbols.stream_wait_event, 5, event, 0),
            (symbols.event_destroy, event),
        ]
        assert api.live_events == set()

    def test_create_failure_raises_stream_error_and_stops(self, h: Harness) -> None:
        runtime, api = _runtime(h)
        api.fail[h.platform.symbols.event_create_with_flags] = 999
        with pytest.raises(StreamError, match=f"fake {h.name} error 999"):
            runtime.make_stream_wait(5, h.stream_cls(h.device0, 3, api))
        assert api.calls == [
            (h.platform.symbols.event_create_with_flags, h.platform.event_disable_timing)
        ]

    def test_record_failure_raises_stream_error_and_still_destroys(self, h: Harness) -> None:
        runtime, api = _runtime(h)
        api.fail[h.platform.symbols.event_record] = 999
        with pytest.raises(StreamError, match=h.platform.symbols.event_record):
            runtime.make_stream_wait(5, h.stream_cls(h.device0, 3, api))
        assert api.calls[-1][0] == h.platform.symbols.event_destroy
        assert api.live_events == set()

    def test_wait_failure_raises_stream_error_and_still_destroys(self, h: Harness) -> None:
        runtime, api = _runtime(h)
        api.fail[h.platform.symbols.stream_wait_event] = 999
        with pytest.raises(StreamError, match=h.platform.symbols.stream_wait_event):
            runtime.make_stream_wait(5, h.stream_cls(h.device0, 3, api))
        assert api.calls[-1][0] == h.platform.symbols.event_destroy

    def test_destroy_failure_raises_stream_error(self, h: Harness) -> None:
        runtime, api = _runtime(h)
        api.fail[h.platform.symbols.event_destroy] = 999
        with pytest.raises(StreamError, match=h.platform.symbols.event_destroy):
            runtime.make_stream_wait(5, h.stream_cls(h.device0, 3, api))


class TestMemcpy:
    @pytest.mark.parametrize(
        "kind",
        [
            CopyKind.HOST_TO_HOST,
            CopyKind.HOST_TO_DEVICE,
            CopyKind.DEVICE_TO_HOST,
            CopyKind.DEFAULT,
        ],
    )
    def test_host_involving_kinds_synchronize_the_stream(self, h: Harness, kind: CopyKind) -> None:
        runtime, api = _runtime(h)
        stream = h.stream_cls(h.device0, 7, api)
        runtime.memcpy(0x1000, 0x2000, 64, kind, stream)
        symbols = h.platform.symbols
        assert api.calls == [
            (symbols.memcpy_async, 0x1000, 0x2000, 64, int(kind), 7),
            (symbols.stream_synchronize, 7),
        ]

    def test_device_to_device_stays_stream_ordered(self, h: Harness) -> None:
        runtime, api = _runtime(h)
        stream = h.stream_cls(h.device0, 7, api)
        runtime.memcpy(0x1000, 0x2000, 64, CopyKind.DEVICE_TO_DEVICE, stream)
        assert api.calls == [(h.platform.symbols.memcpy_async, 0x1000, 0x2000, 64, 3, 7)]

    def test_zero_bytes_is_a_noop(self, h: Harness) -> None:
        runtime, api = _runtime(h)
        runtime.memcpy(
            0x1000, 0x2000, 0, CopyKind.DEVICE_TO_DEVICE, h.stream_cls(h.device0, 7, api)
        )
        assert api.calls == []

    def test_negative_sizes_are_rejected(self, h: Harness) -> None:
        runtime, api = _runtime(h)
        with pytest.raises(ValueError, match="-1"):
            runtime.memcpy(0, 0, -1, CopyKind.DEVICE_TO_DEVICE, h.stream_cls(h.device0, 7, api))

    def test_copy_failure_raises_the_platform_error(self, h: Harness) -> None:
        runtime, api = _runtime(h)
        api.fail[h.platform.symbols.memcpy_async] = 999
        with pytest.raises(h.error_cls, match=h.platform.symbols.memcpy_async):
            runtime.memcpy(
                0x1000, 0x2000, 64, CopyKind.DEVICE_TO_DEVICE, h.stream_cls(h.device0, 7, api)
            )


class TestActivateDevice:
    def test_flips_and_restores_the_native_device(self, h: Harness) -> None:
        runtime, api = _runtime(h, current_device=0)
        with runtime.activate_device(h.device1):
            assert api.current_device == 1
        assert api.current_device == 0
        symbols = h.platform.symbols
        assert api.calls == [
            (symbols.get_device,),
            (symbols.set_device, 1),
            (symbols.set_device, 0),
        ]

    def test_restores_the_previous_device_on_exception(self, h: Harness) -> None:
        runtime, api = _runtime(h, current_device=0)
        with pytest.raises(RuntimeError, match="boom"), runtime.activate_device(h.device1):
            raise RuntimeError("boom")
        assert api.current_device == 0
        assert api.calls[-1] == (h.platform.symbols.set_device, 0)

    def test_skips_the_flip_when_already_current(self, h: Harness) -> None:
        runtime, api = _runtime(h, current_device=1)
        with runtime.activate_device(h.device1):
            pass
        assert api.calls == [(h.platform.symbols.get_device,)]

    def test_get_device_failure_raises_the_platform_error(self, h: Harness) -> None:
        runtime, api = _runtime(h)
        api.fail[h.platform.symbols.get_device] = 999
        with (
            pytest.raises(h.error_cls, match=h.platform.symbols.get_device),
            runtime.activate_device(h.device1),
        ):
            pass

    def test_set_device_failure_raises_the_platform_error(self, h: Harness) -> None:
        runtime, api = _runtime(h)
        api.fail[h.platform.symbols.set_device] = 999
        with (
            pytest.raises(h.error_cls, match=h.platform.symbols.set_device),
            runtime.activate_device(h.device1),
        ):
            pass

    def test_a_failed_restore_never_masks_the_body(self, h: Harness) -> None:
        runtime, api = _runtime(h, current_device=0)
        with runtime.activate_device(h.device1):
            api.fail[h.platform.symbols.set_device] = 999
        # The restore was attempted and its failure swallowed.
        assert api.calls[-1] == (h.platform.symbols.set_device, 0)

    def test_rejects_cpu_devices(self, h: Harness) -> None:
        runtime, _ = _runtime(h)
        with pytest.raises(ValueError, match="cpu"):
            runtime.activate_device(_CPU)


class TestDefaultMemoryResource:
    def test_without_rmm_the_default_is_the_runtime_mr(
        self, h: Harness, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(h.runtime_module, "_import_rmm", lambda: None)
        runtime, api = _runtime(h)
        mr = runtime.default_memory_resource(h.device1)
        assert isinstance(mr, h.raw_mr_cls)
        assert mr.device == h.device1
        assert mr._api is api  # type: ignore[attr-defined]
        # async_alloc="auto": the driver capability was probed.
        assert (
            h.platform.symbols.get_device_attribute,
            h.platform.memory_pools_attribute,
            1,
        ) in api.calls

    def test_with_rmm_the_default_wraps_the_per_device_resource(
        self, h: Harness, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        inner = FakeRmmMemoryResource()
        requested: list[int] = []

        def get_per_device_resource(index: int) -> FakeRmmMemoryResource:
            requested.append(index)
            return inner

        fake_rmm = types.SimpleNamespace(
            mr=types.SimpleNamespace(get_per_device_resource=get_per_device_resource)
        )
        monkeypatch.setattr(h.runtime_module, "_import_rmm", lambda: fake_rmm)
        runtime, _ = _runtime(h)
        mr = runtime.default_memory_resource(h.device1)
        assert isinstance(mr, h.rmm_mr_cls)
        assert mr.inner is inner  # type: ignore[attr-defined]
        assert mr.device == h.device1
        assert requested == [1]

    def test_rejects_cpu_devices(self, h: Harness) -> None:
        runtime, _ = _runtime(h)
        with pytest.raises(ValueError, match="cpu"):
            runtime.default_memory_resource(_CPU)


class TestRmmDisambiguation:
    """hipMM installs a module also named `rmm`, so the name identifies
    nothing; the `rmm.mr` resource-class surface does (design §4.2)."""

    def test_marker_classes_identify_the_platform(self) -> None:
        assert _gpulib.rmm_module_platform(_rmm_module("CudaMemoryResource")) == "cuda"
        assert _gpulib.rmm_module_platform(_rmm_module("CudaAsyncMemoryResource")) == "cuda"
        assert _gpulib.rmm_module_platform(_rmm_module("HipMemoryResource")) == "rocm"
        assert _gpulib.rmm_module_platform(_rmm_module("HipAsyncMemoryResource")) == "rocm"

    def test_ambiguous_or_markerless_modules_identify_nothing(self) -> None:
        assert _gpulib.rmm_module_platform(_rmm_module()) is None
        assert (
            _gpulib.rmm_module_platform(_rmm_module("CudaMemoryResource", "HipMemoryResource"))
            is None
        )
        assert _gpulib.rmm_module_platform(types.ModuleType("rmm")) is None

    def test_import_rmm_accepts_the_matching_platform(
        self, h: Harness, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        module = _rmm_module(h.rmm_marker)
        monkeypatch.setitem(sys.modules, "rmm", module)
        assert h.runtime_module._import_rmm() is module

    def test_import_rmm_rejects_the_sibling_platforms_module(
        self, h: Harness, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        other = HARNESSES["rocm" if h.name == "cuda" else "cuda"]
        monkeypatch.setitem(sys.modules, "rmm", _rmm_module(other.rmm_marker))
        assert h.runtime_module._import_rmm() is None

    def test_import_rmm_without_rmm_returns_none(
        self, h: Harness, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A None entry makes `import rmm` raise ImportError, exactly like an
        # uninstalled module.
        monkeypatch.setitem(sys.modules, "rmm", None)
        assert h.runtime_module._import_rmm() is None

    def test_default_mr_falls_back_when_rmm_targets_the_sibling_platform(
        self, h: Harness, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        other = HARNESSES["rocm" if h.name == "cuda" else "cuda"]
        monkeypatch.setitem(sys.modules, "rmm", _rmm_module(other.rmm_marker))
        runtime, _ = _runtime(h)
        assert isinstance(runtime.default_memory_resource(h.device0), h.raw_mr_cls)

    def test_default_mr_wraps_a_platform_matching_rmm(
        self, h: Harness, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        inner = FakeRmmMemoryResource()
        module = _rmm_module(h.rmm_marker)
        module.mr.get_per_device_resource = lambda index: inner
        monkeypatch.setitem(sys.modules, "rmm", module)
        runtime, _ = _runtime(h)
        mr = runtime.default_memory_resource(h.device0)
        assert isinstance(mr, h.rmm_mr_cls)
        assert mr.inner is inner  # type: ignore[attr-defined]

    def test_env_override_wins_over_a_foreign_rmm_module(
        self, h: Harness, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The pathological §4.2 case: the *other* platform's rmm module is
        # installed, yet DEVMM_RUNTIME still forces this platform's runtime
        # and its default-MR chain falls back to the raw runtime MR.
        other = HARNESSES["rocm" if h.name == "cuda" else "cuda"]
        monkeypatch.setitem(sys.modules, "rmm", _rmm_module(other.rmm_marker))
        api = h.api()
        monkeypatch.setattr(_gpulib, "load_native_api", lambda platform: api)
        monkeypatch.setenv("DEVMM_RUNTIME", h.name)
        runtime = runtime_for(h.device0)
        assert isinstance(runtime, h.runtime_cls)
        assert isinstance(runtime.default_memory_resource(h.device0), h.raw_mr_cls)


class TestGpuDiscovery:
    def test_specs_are_registered_in_platform_order(self) -> None:
        names = [spec.name for spec in _discovery._discovered_specs()]
        assert names.index("cpu") < names.index("cuda") < names.index("rocm")

    def test_probe_passes_when_the_platform_driver_loads(
        self, h: Harness, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        libraries = set(h.driver_libraries)
        monkeypatch.setattr(
            _discovery, "_dlopen", lambda name: object() if name in libraries else None
        )
        names = runtime_names()
        assert h.name in names
        other = "rocm" if h.name == "cuda" else "cuda"
        assert other not in names

    def test_probe_fails_when_no_driver_library_loads(
        self, h: Harness, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(_discovery, "_dlopen", lambda name: None)
        assert h.name not in runtime_names()

    def test_unloadable_runtime_is_skipped_with_an_actionable_message(
        self, h: Harness, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The driver probe passes but the runtime library is absent: the
        # loader raises RuntimeUnavailableError, other runtimes stay usable,
        # and the actionable no-runtime message surfaces (design §4.1).
        libraries = set(h.driver_libraries)
        monkeypatch.setattr(
            _discovery, "_dlopen", lambda name: object() if name in libraries else None
        )
        monkeypatch.setattr(_gpulib, "load_first_library", lambda names: None)
        assert [runtime.name for runtime in available_runtimes()] == ["cpu"]
        with pytest.raises(RuntimeUnavailableError, match=f"supports {h.name}"):
            runtime_for(h.device0)

    def test_env_override_forces_the_runtime(
        self, h: Harness, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        api = h.api()
        monkeypatch.setattr(_gpulib, "load_native_api", lambda platform: api)
        monkeypatch.setenv("DEVMM_RUNTIME", h.name)
        assert runtime_names() == (h.name,)
        runtime = runtime_for(h.device0)
        assert isinstance(runtime, h.runtime_cls)
        assert runtime.api is api

    def test_runtime_for_failure_labels_the_spec_list_registered(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DEVMM_RUNTIME", "cpu")
        with pytest.raises(RuntimeUnavailableError, match="registered: cpu"):
            runtime_for("rocm:0")

    def test_loaded_runtime_resolves_platform_devices(
        self, h: Harness, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        runtime = _install_fake_runtime(monkeypatch, h, h.api())
        assert runtime_for(h.device0) is runtime
        assert runtime_for(h.platform.device_type) is runtime


class TestRuntimeDefaultPath:
    def test_empty_uses_the_runtime_default_stream(
        self, h: Harness, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fake_runtime(monkeypatch, h, h.api())
        t = empty((2,), "float32", device=h.device0, mr=RecordingMemoryResource(device=h.device0))
        stream = t.buffer.stream
        assert isinstance(stream, h.stream_cls)
        assert stream.handle == 0
        assert stream.device == h.device0

    @pytest.mark.usefixtures("_isolated_registry")
    def test_registry_default_resolves_through_the_runtime(
        self, h: Harness, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(h.runtime_module, "_import_rmm", lambda: None)
        _install_fake_runtime(monkeypatch, h, h.api())
        mr = get_current_memory_resource(h.device0)
        assert isinstance(mr, h.raw_mr_cls)
        assert mr.device == h.device0
        assert get_current_memory_resource(h.device0) is mr
