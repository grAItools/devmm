"""CUDA memory resources over fakes (design §5.2, §9): exact alloc/free call
sequences, async-path selection, status->exception mapping and bookkeeping
for `CudaRuntimeMemoryResource`; forwarding, `__cuda_stream__` translation
and the strong wrapped-MR reference for `RmmMemoryResource`.
"""

from __future__ import annotations

import gc
import importlib.util
import weakref
from typing import Any

import pytest

from devmm import Device
from devmm._runtimes import cuda as cuda_module
from devmm._runtimes.base import RuntimeUnavailableError
from devmm._runtimes.cuda import CudaError, CudaStream
from devmm.mrs import cuda as mrs_cuda
from devmm.mrs.cuda import CudaRuntimeMemoryResource, RmmMemoryResource
from tests._cuda_fakes import FakeCudartApi, FakeRmmMemoryResource, FakeRmmStream

_CPU = Device.from_string("cpu")
_CUDA0 = Device.from_string("cuda:0")
_CUDA1 = Device.from_string("cuda:1")

_ATTRIBUTE_PROBE = (
    "cudaDeviceGetAttribute",
    cuda_module.CUDA_DEV_ATTR_MEMORY_POOLS_SUPPORTED,
    1,
)


def _mr(
    async_alloc: bool | str, **api_kwargs: int | bool
) -> tuple[CudaRuntimeMemoryResource, FakeCudartApi]:
    api = FakeCudartApi(**api_kwargs)  # type: ignore[arg-type]
    mr = CudaRuntimeMemoryResource(_CUDA1, async_alloc=async_alloc, api=api)  # type: ignore[arg-type]
    return mr, api


def _stream(api: FakeCudartApi, handle: int = 0x7000, device: Device = _CUDA1) -> CudaStream:
    return CudaStream(device, handle, api)


class TestSyncPath:
    def test_alloc_free_call_sequence_is_exact(self) -> None:
        mr, api = _mr(False, current_device=0)
        stream = _stream(api)
        ptr = mr.allocate(64, stream)
        assert api.calls == [
            ("cudaGetDevice",),
            ("cudaSetDevice", 1),
            ("cudaMalloc", 64),
            ("cudaSetDevice", 0),
        ]
        api.calls.clear()
        mr.deallocate(ptr, 64, stream)
        assert api.calls == [
            ("cudaGetDevice",),
            ("cudaSetDevice", 1),
            ("cudaFree", ptr),
            ("cudaSetDevice", 0),
        ]
        assert api.live_allocations == {}

    def test_no_device_flip_when_already_current(self) -> None:
        mr, api = _mr(False, current_device=1)
        ptr = mr.allocate(64, _stream(api))
        assert api.calls == [("cudaGetDevice",), ("cudaMalloc", 64)]
        api.calls.clear()
        mr.deallocate(ptr, 64, _stream(api))
        assert api.calls == [("cudaGetDevice",), ("cudaFree", ptr)]

    def test_sync_path_is_not_stream_ordered(self) -> None:
        mr, _ = _mr(False)
        assert mr.stream_ordered is False


class TestAsyncSelection:
    def test_auto_probes_and_selects_async_when_supported(self) -> None:
        mr, api = _mr("auto", memory_pools_supported=True, current_device=1)
        assert api.calls == [_ATTRIBUTE_PROBE]
        assert mr.stream_ordered is True
        api.calls.clear()
        stream = _stream(api)
        ptr = mr.allocate(32, stream)
        assert api.calls == [("cudaGetDevice",), ("cudaMallocAsync", 32, 0x7000)]
        api.calls.clear()
        mr.deallocate(ptr, 32, stream)
        assert api.calls == [("cudaGetDevice",), ("cudaFreeAsync", ptr, 0x7000)]

    def test_auto_falls_back_to_sync_when_unsupported(self) -> None:
        mr, api = _mr("auto", memory_pools_supported=False)
        assert api.calls == [_ATTRIBUTE_PROBE]
        assert mr.stream_ordered is False
        mr.allocate(32, _stream(api))
        assert ("cudaMalloc", 32) in api.calls

    def test_auto_treats_a_failed_probe_as_unsupported(self) -> None:
        api = FakeCudartApi(memory_pools_supported=True)
        api.fail["cudaDeviceGetAttribute"] = 999
        mr = CudaRuntimeMemoryResource(_CUDA1, async_alloc="auto", api=api)
        assert mr.stream_ordered is False

    def test_true_forces_async_without_probing(self) -> None:
        mr, api = _mr(True, memory_pools_supported=False)
        assert api.calls == []
        assert mr.stream_ordered is True
        stream = _stream(api)
        mr.allocate(8, stream)
        assert ("cudaMallocAsync", 8, 0x7000) in api.calls

    def test_false_forces_sync_without_probing(self) -> None:
        mr, api = _mr(False, memory_pools_supported=True)
        assert api.calls == []
        assert mr.stream_ordered is False

    def test_other_values_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="async_alloc"):
            CudaRuntimeMemoryResource(_CUDA1, async_alloc="always", api=FakeCudartApi())  # type: ignore[arg-type]


class TestErrorMapping:
    def test_sync_allocation_failure_raises_memory_error(self) -> None:
        mr, api = _mr(False)
        api.fail["cudaMalloc"] = 2
        with pytest.raises(MemoryError, match="fake cudart error 2") as excinfo:
            mr.allocate(64, _stream(api))
        assert "cuda:1" in str(excinfo.value)
        assert mr._debug_live_count() == 0

    def test_async_allocation_failure_raises_memory_error(self) -> None:
        mr, api = _mr(True)
        api.fail["cudaMallocAsync"] = 2
        with pytest.raises(MemoryError, match="fake cudart error 2"):
            mr.allocate(64, _stream(api))

    def test_free_failure_raises_cuda_error(self) -> None:
        mr, api = _mr(False)
        stream = _stream(api)
        ptr = mr.allocate(64, stream)
        api.fail["cudaFree"] = 999
        with pytest.raises(CudaError, match="cudaFree"):
            mr.deallocate(ptr, 64, stream)

    def test_double_free_raises_before_touching_the_api(self) -> None:
        mr, api = _mr(False)
        stream = _stream(api)
        ptr = mr.allocate(64, stream)
        mr.deallocate(ptr, 64, stream)
        api.calls.clear()
        with pytest.raises(ValueError, match="live allocation"):
            mr.deallocate(ptr, 64, stream)
        assert api.calls == []

    def test_foreign_pointer_free_raises(self) -> None:
        mr, api = _mr(False)
        stream = _stream(api)
        ptr = mr.allocate(64, stream)
        with pytest.raises(ValueError, match="live allocation"):
            mr.deallocate(ptr + 1, 64, stream)
        mr.deallocate(ptr, 64, stream)
        assert mr._debug_live_count() == 0

    def test_size_mismatched_free_raises_and_keeps_the_allocation_live(self) -> None:
        mr, api = _mr(False)
        stream = _stream(api)
        ptr = mr.allocate(64, stream)
        with pytest.raises(ValueError, match="size mismatch"):
            mr.deallocate(ptr, 63, stream)
        assert mr._debug_live_count() == 1
        mr.deallocate(ptr, 64, stream)

    def test_negative_allocation_size_is_rejected_without_api_calls(self) -> None:
        mr, api = _mr(False)
        api.calls.clear()
        with pytest.raises(ValueError, match="-1"):
            mr.allocate(-1, _stream(api))
        assert api.calls == []

    def test_non_cuda_devices_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="cpu"):
            CudaRuntimeMemoryResource(_CPU, api=FakeCudartApi())


class TestContracts:
    def test_zero_byte_allocations_get_unique_freeable_pointers(self) -> None:
        mr, api = _mr(False)
        stream = _stream(api)
        first = mr.allocate(0, stream)
        second = mr.allocate(0, stream)
        assert first != 0
        assert second != 0
        assert first != second
        # The driver never sees a zero-byte request (cudaMalloc(0) returns
        # NULL, which could not be tracked or freed).
        assert ("cudaMalloc", 1) in api.calls
        mr.deallocate(first, 0, stream)
        mr.deallocate(second, 0, stream)
        assert mr._debug_live_count() == 0

    def test_guaranteed_alignment_is_256(self) -> None:
        mr, _ = _mr(False)
        assert mr.guaranteed_alignment() == 256

    def test_live_count_tracks_outstanding_allocations(self) -> None:
        mr, api = _mr(False)
        stream = _stream(api)
        ptrs = [mr.allocate(nbytes, stream) for nbytes in (16, 32, 64)]
        assert mr._debug_live_count() == 3
        for ptr, nbytes in zip(ptrs, (16, 32, 64), strict=True):
            mr.deallocate(ptr, nbytes, stream)
        assert mr._debug_live_count() == 0

    def test_repr_names_the_device_and_mode(self) -> None:
        mr, _ = _mr(True)
        assert "cuda" in repr(mr)


@pytest.fixture
def _fake_rmm_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mrs_cuda, "_rmm_stream_class", lambda: FakeRmmStream)


@pytest.mark.usefixtures("_fake_rmm_stream")
class TestRmmMemoryResource:
    def test_forwards_allocate_with_a_translated_stream(self) -> None:
        inner = FakeRmmMemoryResource()
        mr = RmmMemoryResource(inner, _CUDA0)
        stream = CudaStream(_CUDA0, 0xABC, FakeCudartApi())
        mr.allocate(64, stream)
        name, nbytes, translated = inner.calls[0]
        assert (name, nbytes) == ("allocate", 64)
        assert isinstance(translated, FakeRmmStream)
        # Translation rides the CUDA stream protocol: rmm's Stream reads the
        # wrapped object's __cuda_stream__ (design §5.2).
        assert translated.wrapped is stream
        assert stream.__cuda_stream__() == (0, 0xABC)

    def test_forwards_deallocate_with_a_translated_stream(self) -> None:
        inner = FakeRmmMemoryResource()
        mr = RmmMemoryResource(inner, _CUDA0)
        stream = CudaStream(_CUDA0, 0xABC, FakeCudartApi())
        ptr = mr.allocate(64, stream)
        mr.deallocate(ptr, 64, stream)
        name, freed_ptr, nbytes, translated = inner.calls[1]
        assert (name, freed_ptr, nbytes) == ("deallocate", ptr, 64)
        assert isinstance(translated, FakeRmmStream)
        assert translated.wrapped is stream

    def test_zero_byte_requests_are_bumped_to_one_byte_both_ways(self) -> None:
        inner = FakeRmmMemoryResource()
        mr = RmmMemoryResource(inner, _CUDA0)
        stream = CudaStream(_CUDA0, 0, FakeCudartApi())
        first = mr.allocate(0, stream)
        second = mr.allocate(0, stream)
        assert first != second
        mr.deallocate(first, 0, stream)
        assert inner.calls[0][1] == 1
        assert inner.calls[2][2] == 1

    def test_holds_the_wrapped_mr_strongly(self) -> None:
        inner = FakeRmmMemoryResource()
        ref = weakref.ref(inner)
        mr = RmmMemoryResource(inner, _CUDA0)
        del inner
        gc.collect()
        assert ref() is not None
        assert mr.inner is ref()

    def test_capability_probes(self) -> None:
        mr = RmmMemoryResource(FakeRmmMemoryResource(), _CUDA0)
        assert mr.stream_ordered is True
        assert mr.guaranteed_alignment() == 256
        assert mr.device == _CUDA0

    def test_rejects_non_cuda_devices(self) -> None:
        with pytest.raises(ValueError, match="cpu"):
            RmmMemoryResource(FakeRmmMemoryResource(), _CPU)

    def test_double_free_raises_without_forwarding(self) -> None:
        inner = FakeRmmMemoryResource()
        mr = RmmMemoryResource(inner, _CUDA0)
        stream = CudaStream(_CUDA0, 0, FakeCudartApi())
        ptr = mr.allocate(64, stream)
        mr.deallocate(ptr, 64, stream)
        calls_before = list(inner.calls)
        with pytest.raises(ValueError, match="live allocation"):
            mr.deallocate(ptr, 64, stream)
        assert inner.calls == calls_before

    def test_size_mismatched_free_raises_and_keeps_the_allocation_live(self) -> None:
        inner = FakeRmmMemoryResource()
        mr = RmmMemoryResource(inner, _CUDA0)
        stream = CudaStream(_CUDA0, 0, FakeCudartApi())
        ptr = mr.allocate(64, stream)
        with pytest.raises(ValueError, match="size mismatch"):
            mr.deallocate(ptr, 32, stream)
        assert mr._debug_live_count() == 1

    def test_inner_allocation_failures_propagate_untouched(self) -> None:
        marker = MemoryError("pool exhausted")

        class _ExplodingInner:
            def allocate(self, nbytes: int, stream: Any) -> int:
                raise marker

            def deallocate(self, ptr: int, nbytes: int, stream: Any) -> None:
                raise AssertionError("never reached")

        mr = RmmMemoryResource(_ExplodingInner(), _CUDA0)
        stream = CudaStream(_CUDA0, 0, FakeCudartApi())
        with pytest.raises(MemoryError) as excinfo:
            mr.allocate(64, stream)
        assert excinfo.value is marker
        assert mr._debug_live_count() == 0


@pytest.mark.skipif(importlib.util.find_spec("rmm") is not None, reason="rmm is installed here")
def test_stream_translation_without_rmm_raises_runtime_unavailable() -> None:
    with pytest.raises(RuntimeUnavailableError, match="rmm"):
        mrs_cuda._rmm_stream_class()
