"""The CUDA `DeviceRuntime` over a scripted `FakeCudartApi` (design §4, §9):
call sequences and failure injection for streams, memcpy, device activation
and the DLPack handoff primitive, plus the driver-keyed discovery wiring and
the runtime-backed default stream/default MR for `empty()` — no hardware.
"""

from __future__ import annotations

import ctypes
import gc
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
from devmm._core.stream import (
    DEFAULT,
    LEGACY_DEFAULT,
    PER_THREAD_DEFAULT,
    CpuStream,
    StreamError,
)
from devmm._runtimes import _discovery
from devmm._runtimes import cuda as cuda_module
from devmm._runtimes.base import CopyKind, DeviceRuntime, RuntimeUnavailableError
from devmm._runtimes.cuda import CudaError, CudaRuntime, CudaStream
from devmm.mrs.cuda import CudaRuntimeMemoryResource, RmmMemoryResource
from devmm.testing import RecordingMemoryResource
from tests._cuda_fakes import AsynclessLibcudart, FakeCudartApi, FakeRmmMemoryResource

_CPU = Device.from_string("cpu")
_CUDA0 = Device.from_string("cuda:0")
_CUDA1 = Device.from_string("cuda:1")

# Structural SPI conformance, verified by mypy --strict: the CUDA runtime
# satisfies the `DeviceRuntime` protocol.
_SPI_CHECK: DeviceRuntime = CudaRuntime(api=FakeCudartApi())


def _runtime(**api_kwargs: int | bool) -> tuple[CudaRuntime, FakeCudartApi]:
    api = FakeCudartApi(**api_kwargs)  # type: ignore[arg-type]
    return CudaRuntime(api=api), api


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


def _install_fake_cuda_runtime(monkeypatch: pytest.MonkeyPatch, api: FakeCudartApi) -> CudaRuntime:
    """Replace the built-in cuda spec with one resolving to a fake-api runtime."""
    runtime = CudaRuntime(api=api)
    specs = tuple(
        _discovery._RuntimeSpec("cuda", lambda: True, lambda: runtime)
        if spec.name == "cuda"
        else spec
        for spec in _discovery._BUILTIN_SPECS
    )
    monkeypatch.setattr(_discovery, "_BUILTIN_SPECS", specs)
    _discovery._clear_spec_cache()
    return runtime


class TestIdentity:
    def test_identity(self) -> None:
        runtime, _ = _runtime()
        assert runtime.name == "cuda"
        assert runtime.device_types == frozenset({DeviceType.CUDA})

    def test_device_count_queries_the_api(self) -> None:
        runtime, api = _runtime(device_count=3)
        assert runtime.device_count(DeviceType.CUDA) == 3
        assert api.calls == [("cudaGetDeviceCount",)]

    def test_device_count_is_zero_for_foreign_device_types(self) -> None:
        runtime, api = _runtime()
        assert runtime.device_count(DeviceType.CPU) == 0
        assert runtime.device_count(DeviceType.ROCM) == 0
        assert api.calls == []

    def test_device_count_failure_raises_cuda_error(self) -> None:
        runtime, api = _runtime()
        api.fail["cudaGetDeviceCount"] = 999
        with pytest.raises(CudaError, match="fake cudart error 999"):
            runtime.device_count(DeviceType.CUDA)

    def test_unloadable_libcudart_raises_runtime_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(cuda_module, "_load_first_library", lambda names: None)
        with pytest.raises(RuntimeUnavailableError, match="libcudart"):
            CudaRuntime()


class TestLibcudartApi:
    def test_missing_async_symbols_report_not_supported(self) -> None:
        # A pre-11.2 libcudart lacks the cudaMallocAsync/cudaFreeAsync pair;
        # the api shim constructs anyway and the async family reports
        # cudaErrorNotSupported instead of raising AttributeError.
        api = cuda_module._LibcudartApi(cast(ctypes.CDLL, AsynclessLibcudart()))
        assert api.cudaMallocAsync(64, 0) == (cuda_module.CUDA_ERROR_NOT_SUPPORTED, 0)
        assert api.cudaFreeAsync(0x1000, 0) == cuda_module.CUDA_ERROR_NOT_SUPPORTED


class TestStreams:
    def test_default_stream_is_the_default_handle(self) -> None:
        runtime, api = _runtime()
        stream = runtime.default_stream(_CUDA1)
        assert isinstance(stream, CudaStream)
        assert stream.handle == 0
        assert stream.device == _CUDA1
        assert api.calls == []

    def test_create_stream_activates_the_device_and_creates(self) -> None:
        runtime, api = _runtime(current_device=0)
        stream = runtime.create_stream(_CUDA1)
        assert api.calls == [
            ("cudaGetDevice",),
            ("cudaSetDevice", 1),
            ("cudaStreamCreate",),
            ("cudaSetDevice", 0),
        ]
        assert stream.device == _CUDA1
        assert stream.handle in api.live_streams

    def test_created_stream_is_destroyed_with_its_wrapper(self) -> None:
        runtime, api = _runtime()
        stream = runtime.create_stream(_CUDA0)
        handle = stream.handle
        del stream
        gc.collect()
        assert ("cudaStreamDestroy", handle) in api.calls
        assert handle not in api.live_streams

    def test_create_stream_failure_raises_cuda_error(self) -> None:
        runtime, api = _runtime()
        api.fail["cudaStreamCreate"] = 999
        with pytest.raises(CudaError, match="cudaStreamCreate"):
            runtime.create_stream(_CUDA0)

    def test_synchronize_routes_through_the_api(self) -> None:
        _, api = _runtime()
        stream = CudaStream(_CUDA0, 7, api)
        stream.synchronize()
        assert api.calls == [("cudaStreamSynchronize", 7)]

    def test_synchronize_failure_raises_cuda_error(self) -> None:
        _, api = _runtime()
        api.fail["cudaStreamSynchronize"] = 999
        with pytest.raises(CudaError, match="cudaStreamSynchronize"):
            CudaStream(_CUDA0, 7, api).synchronize()

    def test_wait_raw_orders_via_an_event(self) -> None:
        _, api = _runtime()
        stream = CudaStream(_CUDA0, 7, api)
        stream.wait_raw(9)
        event = api.calls[1][1]
        assert api.calls == [
            ("cudaEventCreateWithFlags", cuda_module.CUDA_EVENT_DISABLE_TIMING),
            ("cudaEventRecord", event, 9),
            ("cudaStreamWaitEvent", 7, event, 0),
            ("cudaEventDestroy", event),
        ]
        assert api.live_events == set()

    def test_cuda_stream_implements_the_cuda_stream_protocol(self) -> None:
        _, api = _runtime()
        assert CudaStream(_CUDA0, 0xBEEF, api).__cuda_stream__() == (0, 0xBEEF)

    def test_cuda_stream_rejects_non_cuda_devices(self) -> None:
        _, api = _runtime()
        with pytest.raises(ValueError, match="cpu"):
            CudaStream(_CPU, 0, api)

    def test_cuda_stream_rejects_negative_handles(self) -> None:
        _, api = _runtime()
        with pytest.raises(ValueError, match="-1"):
            CudaStream(_CUDA0, -1, api)


class TestWrapStream:
    def test_passes_through_a_stream_on_the_same_device(self) -> None:
        runtime, api = _runtime()
        stream = CudaStream(_CUDA0, 5, api)
        assert runtime.wrap_stream(_CUDA0, stream) is stream

    def test_rejects_streams_on_other_devices(self) -> None:
        runtime, _ = _runtime()
        with pytest.raises(ValueError, match="cpu"):
            runtime.wrap_stream(_CUDA0, CpuStream())

    def test_maps_sentinels_to_the_cuda_magic_handles(self) -> None:
        runtime, _ = _runtime()
        for sentinel, handle in ((DEFAULT, 0), (LEGACY_DEFAULT, 1), (PER_THREAD_DEFAULT, 2)):
            stream = runtime.wrap_stream(_CUDA0, sentinel)
            assert isinstance(stream, CudaStream)
            assert stream.handle == handle

    def test_wraps_raw_int_handles(self) -> None:
        runtime, _ = _runtime()
        stream = runtime.wrap_stream(_CUDA1, 0xBEEF)
        assert isinstance(stream, CudaStream)
        assert stream.handle == 0xBEEF
        assert stream.device == _CUDA1

    def test_rejects_negative_handles(self) -> None:
        runtime, _ = _runtime()
        with pytest.raises(ValueError, match="-3"):
            runtime.wrap_stream(_CUDA0, -3)

    def test_accepts_cuda_stream_protocol_objects(self) -> None:
        class _Proto:
            def __cuda_stream__(self) -> tuple[int, int]:
                return (0, 0xF00D)

        runtime, _ = _runtime()
        stream = runtime.wrap_stream(_CUDA0, _Proto())
        assert isinstance(stream, CudaStream)
        assert stream.handle == 0xF00D

    def test_rejects_non_int_protocol_handles(self) -> None:
        class _BadProto:
            def __cuda_stream__(self) -> tuple[int, str]:
                return (0, "seven")

        runtime, _ = _runtime()
        with pytest.raises(TypeError):
            runtime.wrap_stream(_CUDA0, _BadProto())

    def test_rejects_unknown_objects(self) -> None:
        runtime, _ = _runtime()
        with pytest.raises(TypeError):
            runtime.wrap_stream(_CUDA0, object())

    def test_rejects_non_cuda_devices(self) -> None:
        runtime, _ = _runtime()
        with pytest.raises(ValueError, match="cpu"):
            runtime.wrap_stream(_CPU, 0)


class TestMakeStreamWait:
    def test_emits_create_record_wait_destroy(self) -> None:
        runtime, api = _runtime()
        producer = CudaStream(_CUDA0, 3, api)
        runtime.make_stream_wait(5, producer)
        event = api.calls[1][1]
        assert api.calls == [
            ("cudaEventCreateWithFlags", cuda_module.CUDA_EVENT_DISABLE_TIMING),
            ("cudaEventRecord", event, 3),
            ("cudaStreamWaitEvent", 5, event, 0),
            ("cudaEventDestroy", event),
        ]
        assert api.live_events == set()

    def test_create_failure_raises_stream_error_and_stops(self) -> None:
        runtime, api = _runtime()
        api.fail["cudaEventCreateWithFlags"] = 999
        with pytest.raises(StreamError, match="fake cudart error 999"):
            runtime.make_stream_wait(5, CudaStream(_CUDA0, 3, api))
        assert api.calls == [("cudaEventCreateWithFlags", cuda_module.CUDA_EVENT_DISABLE_TIMING)]

    def test_record_failure_raises_stream_error_and_still_destroys(self) -> None:
        runtime, api = _runtime()
        api.fail["cudaEventRecord"] = 999
        with pytest.raises(StreamError, match="cudaEventRecord"):
            runtime.make_stream_wait(5, CudaStream(_CUDA0, 3, api))
        assert api.calls[-1][0] == "cudaEventDestroy"
        assert api.live_events == set()

    def test_wait_failure_raises_stream_error_and_still_destroys(self) -> None:
        runtime, api = _runtime()
        api.fail["cudaStreamWaitEvent"] = 999
        with pytest.raises(StreamError, match="cudaStreamWaitEvent"):
            runtime.make_stream_wait(5, CudaStream(_CUDA0, 3, api))
        assert api.calls[-1][0] == "cudaEventDestroy"

    def test_destroy_failure_raises_stream_error(self) -> None:
        runtime, api = _runtime()
        api.fail["cudaEventDestroy"] = 999
        with pytest.raises(StreamError, match="cudaEventDestroy"):
            runtime.make_stream_wait(5, CudaStream(_CUDA0, 3, api))


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
    def test_host_involving_kinds_synchronize_the_stream(self, kind: CopyKind) -> None:
        runtime, api = _runtime()
        stream = CudaStream(_CUDA0, 7, api)
        runtime.memcpy(0x1000, 0x2000, 64, kind, stream)
        assert api.calls == [
            ("cudaMemcpyAsync", 0x1000, 0x2000, 64, int(kind), 7),
            ("cudaStreamSynchronize", 7),
        ]

    def test_device_to_device_stays_stream_ordered(self) -> None:
        runtime, api = _runtime()
        stream = CudaStream(_CUDA0, 7, api)
        runtime.memcpy(0x1000, 0x2000, 64, CopyKind.DEVICE_TO_DEVICE, stream)
        assert api.calls == [("cudaMemcpyAsync", 0x1000, 0x2000, 64, 3, 7)]

    def test_zero_bytes_is_a_noop(self) -> None:
        runtime, api = _runtime()
        runtime.memcpy(0x1000, 0x2000, 0, CopyKind.DEVICE_TO_DEVICE, CudaStream(_CUDA0, 7, api))
        assert api.calls == []

    def test_negative_sizes_are_rejected(self) -> None:
        runtime, api = _runtime()
        with pytest.raises(ValueError, match="-1"):
            runtime.memcpy(0, 0, -1, CopyKind.DEVICE_TO_DEVICE, CudaStream(_CUDA0, 7, api))

    def test_copy_failure_raises_cuda_error(self) -> None:
        runtime, api = _runtime()
        api.fail["cudaMemcpyAsync"] = 999
        with pytest.raises(CudaError, match="cudaMemcpyAsync"):
            runtime.memcpy(
                0x1000, 0x2000, 64, CopyKind.DEVICE_TO_DEVICE, CudaStream(_CUDA0, 7, api)
            )


class TestActivateDevice:
    def test_flips_and_restores_the_native_device(self) -> None:
        runtime, api = _runtime(current_device=0)
        with runtime.activate_device(_CUDA1):
            assert api.current_device == 1
        assert api.current_device == 0
        assert api.calls == [("cudaGetDevice",), ("cudaSetDevice", 1), ("cudaSetDevice", 0)]

    def test_restores_the_previous_device_on_exception(self) -> None:
        runtime, api = _runtime(current_device=0)
        with pytest.raises(RuntimeError, match="boom"), runtime.activate_device(_CUDA1):
            raise RuntimeError("boom")
        assert api.current_device == 0
        assert api.calls[-1] == ("cudaSetDevice", 0)

    def test_skips_the_flip_when_already_current(self) -> None:
        runtime, api = _runtime(current_device=1)
        with runtime.activate_device(_CUDA1):
            pass
        assert api.calls == [("cudaGetDevice",)]

    def test_get_device_failure_raises_cuda_error(self) -> None:
        runtime, api = _runtime()
        api.fail["cudaGetDevice"] = 999
        with pytest.raises(CudaError, match="cudaGetDevice"), runtime.activate_device(_CUDA1):
            pass

    def test_set_device_failure_raises_cuda_error(self) -> None:
        runtime, api = _runtime()
        api.fail["cudaSetDevice"] = 999
        with pytest.raises(CudaError, match="cudaSetDevice"), runtime.activate_device(_CUDA1):
            pass

    def test_a_failed_restore_never_masks_the_body(self) -> None:
        runtime, api = _runtime(current_device=0)
        with runtime.activate_device(_CUDA1):
            api.fail["cudaSetDevice"] = 999
        # The restore was attempted and its failure swallowed.
        assert api.calls[-1] == ("cudaSetDevice", 0)

    def test_rejects_non_cuda_devices(self) -> None:
        runtime, _ = _runtime()
        with pytest.raises(ValueError, match="cpu"):
            runtime.activate_device(_CPU)


class TestDefaultMemoryResource:
    def test_without_rmm_the_default_is_the_runtime_mr(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(cuda_module, "_import_rmm", lambda: None)
        runtime, api = _runtime()
        mr = runtime.default_memory_resource(_CUDA1)
        assert isinstance(mr, CudaRuntimeMemoryResource)
        assert mr.device == _CUDA1
        assert mr._api is api
        # async_alloc="auto": the driver capability was probed.
        assert (
            "cudaDeviceGetAttribute",
            cuda_module.CUDA_DEV_ATTR_MEMORY_POOLS_SUPPORTED,
            1,
        ) in api.calls

    def test_with_rmm_the_default_wraps_the_per_device_resource(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        inner = FakeRmmMemoryResource()
        requested: list[int] = []

        def get_per_device_resource(index: int) -> FakeRmmMemoryResource:
            requested.append(index)
            return inner

        fake_rmm = types.SimpleNamespace(
            mr=types.SimpleNamespace(get_per_device_resource=get_per_device_resource)
        )
        monkeypatch.setattr(cuda_module, "_import_rmm", lambda: fake_rmm)
        runtime, _ = _runtime()
        mr = runtime.default_memory_resource(_CUDA1)
        assert isinstance(mr, RmmMemoryResource)
        assert mr.inner is inner
        assert mr.device == _CUDA1
        assert requested == [1]

    def test_rejects_non_cuda_devices(self) -> None:
        runtime, _ = _runtime()
        with pytest.raises(ValueError, match="cpu"):
            runtime.default_memory_resource(_CPU)


class TestCudaDiscovery:
    def test_cuda_spec_is_registered_after_cpu(self) -> None:
        names = [spec.name for spec in _discovery._discovered_specs()]
        assert names.index("cpu") < names.index("cuda")

    def test_probe_passes_when_a_driver_library_loads(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(_discovery, "_dlopen", lambda name: object())
        assert "cuda" in runtime_names()

    def test_probe_fails_when_no_driver_library_loads(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(_discovery, "_dlopen", lambda name: None)
        assert "cuda" not in runtime_names()

    def test_unloadable_runtime_is_skipped_with_an_actionable_message(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The driver probe passes but libcudart is absent: the loader raises
        # RuntimeUnavailableError, other runtimes stay usable, and the
        # actionable no-runtime message surfaces (design §4.1).
        monkeypatch.setattr(_discovery, "_dlopen", lambda name: object())
        monkeypatch.setattr(cuda_module, "_load_first_library", lambda names: None)
        assert [runtime.name for runtime in available_runtimes()] == ["cpu"]
        with pytest.raises(RuntimeUnavailableError, match="supports cuda"):
            runtime_for(_CUDA0)

    def test_env_override_forces_the_cuda_runtime(self, monkeypatch: pytest.MonkeyPatch) -> None:
        api = FakeCudartApi()
        monkeypatch.setattr(cuda_module, "_default_api", lambda: api)
        monkeypatch.setenv("DEVMM_RUNTIME", "cuda")
        assert runtime_names() == ("cuda",)
        runtime = runtime_for(_CUDA0)
        assert isinstance(runtime, CudaRuntime)
        assert runtime.api is api

    def test_runtime_for_failure_labels_the_spec_list_registered(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DEVMM_RUNTIME", "cpu")
        with pytest.raises(RuntimeUnavailableError, match="registered: cpu"):
            runtime_for("rocm:0")

    def test_loaded_runtime_resolves_cuda_devices(self, monkeypatch: pytest.MonkeyPatch) -> None:
        api = FakeCudartApi()
        runtime = _install_fake_cuda_runtime(monkeypatch, api)
        assert runtime_for(_CUDA0) is runtime
        assert runtime_for(DeviceType.CUDA) is runtime


class TestRuntimeDefaultPath:
    def test_empty_uses_the_runtime_default_stream(self, monkeypatch: pytest.MonkeyPatch) -> None:
        api = FakeCudartApi()
        _install_fake_cuda_runtime(monkeypatch, api)
        t = empty((2,), "float32", device=_CUDA0, mr=RecordingMemoryResource(device=_CUDA0))
        stream = t.buffer.stream
        assert isinstance(stream, CudaStream)
        assert stream.handle == 0
        assert stream.device == _CUDA0

    @pytest.mark.usefixtures("_isolated_registry")
    def test_registry_default_resolves_through_the_cuda_runtime(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(cuda_module, "_import_rmm", lambda: None)
        api = FakeCudartApi()
        _install_fake_cuda_runtime(monkeypatch, api)
        mr = get_current_memory_resource(_CUDA0)
        assert isinstance(mr, CudaRuntimeMemoryResource)
        assert mr.device == _CUDA0
        assert get_current_memory_resource(_CUDA0) is mr
