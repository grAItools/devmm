"""The CUDA and ROCm memory resources over fakes (design §5.2, §5.3, §9),
one shared suite parametrized over both platform harnesses: exact alloc/free
call sequences, async-path selection, status->exception mapping and
bookkeeping for the raw runtime MRs; forwarding, `__cuda_stream__` stream
translation and the strong wrapped-MR reference for the rmm-module wrappers.
"""

from __future__ import annotations

import gc
import importlib.util
import weakref
from typing import Any, cast

import pytest

from devmm import Device
from devmm._core.stream import Stream
from devmm._runtimes._gpulib import GpuRuntimeMemoryResource, RmmLikeMemoryResource
from devmm._runtimes.base import RuntimeUnavailableError
from devmm.mrs import cuda as mrs_cuda
from devmm.mrs import rocm as mrs_rocm
from tests._gpu_fakes import HARNESSES, FakeGpuApi, FakeRmmMemoryResource, FakeRmmStream
from tests._gpu_fakes import GpuHarness as Harness

_CPU = Device.from_string("cpu")


@pytest.fixture(params=sorted(HARNESSES))
def h(request: pytest.FixtureRequest) -> Harness:
    return HARNESSES[cast(str, request.param)]


def _mr(
    h: Harness, async_alloc: bool | str, **api_kwargs: int | bool
) -> tuple[GpuRuntimeMemoryResource, FakeGpuApi]:
    api = h.api(**api_kwargs)
    mr = h.raw_mr_cls(h.device1, async_alloc=async_alloc, api=api)  # type: ignore[call-arg]
    return cast(GpuRuntimeMemoryResource, mr), api


def _stream(h: Harness, api: FakeGpuApi, handle: int = 0x7000) -> Stream:
    return h.stream_cls(h.device1, handle, api)


def _attribute_probe(h: Harness) -> tuple[Any, ...]:
    return (
        h.platform.symbols.get_device_attribute,
        h.platform.memory_pools_attribute,
        1,
    )


class TestSyncPath:
    def test_alloc_free_call_sequence_is_exact(self, h: Harness) -> None:
        mr, api = _mr(h, False, current_device=0)
        stream = _stream(h, api)
        symbols = h.platform.symbols
        ptr = mr.allocate(64, stream)
        assert api.calls == [
            (symbols.get_device,),
            (symbols.set_device, 1),
            (symbols.malloc, 64),
            (symbols.set_device, 0),
        ]
        api.calls.clear()
        mr.deallocate(ptr, 64, stream)
        assert api.calls == [
            (symbols.get_device,),
            (symbols.set_device, 1),
            (symbols.free, ptr),
            (symbols.set_device, 0),
        ]
        assert api.live_allocations == {}

    def test_no_device_flip_when_already_current(self, h: Harness) -> None:
        mr, api = _mr(h, False, current_device=1)
        symbols = h.platform.symbols
        ptr = mr.allocate(64, _stream(h, api))
        assert api.calls == [(symbols.get_device,), (symbols.malloc, 64)]
        api.calls.clear()
        mr.deallocate(ptr, 64, _stream(h, api))
        assert api.calls == [(symbols.get_device,), (symbols.free, ptr)]

    def test_sync_path_is_not_stream_ordered(self, h: Harness) -> None:
        mr, _ = _mr(h, False)
        assert mr.stream_ordered is False


class TestAsyncSelection:
    def test_auto_probes_and_selects_async_when_supported(self, h: Harness) -> None:
        mr, api = _mr(h, "auto", memory_pools_supported=True, current_device=1)
        assert api.calls == [_attribute_probe(h)]
        assert mr.stream_ordered is True
        api.calls.clear()
        stream = _stream(h, api)
        symbols = h.platform.symbols
        ptr = mr.allocate(32, stream)
        assert api.calls == [(symbols.get_device,), (symbols.malloc_async, 32, 0x7000)]
        api.calls.clear()
        mr.deallocate(ptr, 32, stream)
        assert api.calls == [(symbols.get_device,), (symbols.free_async, ptr, 0x7000)]

    def test_auto_falls_back_to_sync_when_unsupported(self, h: Harness) -> None:
        mr, api = _mr(h, "auto", memory_pools_supported=False)
        assert api.calls == [_attribute_probe(h)]
        assert mr.stream_ordered is False
        mr.allocate(32, _stream(h, api))
        assert (h.platform.symbols.malloc, 32) in api.calls

    def test_auto_treats_a_failed_probe_as_unsupported(self, h: Harness) -> None:
        api = h.api(memory_pools_supported=True)
        api.fail[h.platform.symbols.get_device_attribute] = 999
        mr = h.raw_mr_cls(h.device1, async_alloc="auto", api=api)  # type: ignore[call-arg]
        assert mr.stream_ordered is False

    def test_true_forces_async_without_probing(self, h: Harness) -> None:
        mr, api = _mr(h, True, memory_pools_supported=False)
        assert api.calls == []
        assert mr.stream_ordered is True
        stream = _stream(h, api)
        mr.allocate(8, stream)
        assert (h.platform.symbols.malloc_async, 8, 0x7000) in api.calls

    def test_false_forces_sync_without_probing(self, h: Harness) -> None:
        mr, api = _mr(h, False, memory_pools_supported=True)
        assert api.calls == []
        assert mr.stream_ordered is False

    def test_other_values_are_rejected(self, h: Harness) -> None:
        with pytest.raises(ValueError, match="async_alloc"):
            h.raw_mr_cls(h.device1, async_alloc="always", api=h.api())  # type: ignore[call-arg]


class TestErrorMapping:
    def test_sync_allocation_failure_raises_memory_error(self, h: Harness) -> None:
        mr, api = _mr(h, False)
        api.fail[h.platform.symbols.malloc] = 2
        with pytest.raises(MemoryError, match=f"fake {h.name} error 2") as excinfo:
            mr.allocate(64, _stream(h, api))
        assert f"{h.name}:1" in str(excinfo.value)
        assert mr._debug_live_count() == 0

    def test_async_allocation_failure_raises_memory_error(self, h: Harness) -> None:
        mr, api = _mr(h, True)
        api.fail[h.platform.symbols.malloc_async] = 2
        with pytest.raises(MemoryError, match=f"fake {h.name} error 2"):
            mr.allocate(64, _stream(h, api))

    def test_free_failure_raises_the_platform_error(self, h: Harness) -> None:
        mr, api = _mr(h, False)
        stream = _stream(h, api)
        ptr = mr.allocate(64, stream)
        api.fail[h.platform.symbols.free] = 999
        with pytest.raises(h.error_cls, match=h.platform.symbols.free):
            mr.deallocate(ptr, 64, stream)

    def test_double_free_raises_before_touching_the_api(self, h: Harness) -> None:
        mr, api = _mr(h, False)
        stream = _stream(h, api)
        ptr = mr.allocate(64, stream)
        mr.deallocate(ptr, 64, stream)
        api.calls.clear()
        with pytest.raises(ValueError, match="live allocation"):
            mr.deallocate(ptr, 64, stream)
        assert api.calls == []

    def test_foreign_pointer_free_raises(self, h: Harness) -> None:
        mr, api = _mr(h, False)
        stream = _stream(h, api)
        ptr = mr.allocate(64, stream)
        with pytest.raises(ValueError, match="live allocation"):
            mr.deallocate(ptr + 1, 64, stream)
        mr.deallocate(ptr, 64, stream)
        assert mr._debug_live_count() == 0

    def test_size_mismatched_free_raises_and_keeps_the_allocation_live(self, h: Harness) -> None:
        mr, api = _mr(h, False)
        stream = _stream(h, api)
        ptr = mr.allocate(64, stream)
        with pytest.raises(ValueError, match="size mismatch"):
            mr.deallocate(ptr, 63, stream)
        assert mr._debug_live_count() == 1
        mr.deallocate(ptr, 64, stream)

    def test_negative_allocation_size_is_rejected_without_api_calls(self, h: Harness) -> None:
        mr, api = _mr(h, False)
        api.calls.clear()
        with pytest.raises(ValueError, match="-1"):
            mr.allocate(-1, _stream(h, api))
        assert api.calls == []

    def test_cpu_devices_are_rejected(self, h: Harness) -> None:
        with pytest.raises(ValueError, match="cpu"):
            h.raw_mr_cls(_CPU, api=h.api())  # type: ignore[call-arg]

    def test_sibling_platform_devices_are_rejected(self, h: Harness) -> None:
        foreign = Device(h.foreign_device_type, 0)
        with pytest.raises(ValueError, match=h.name):
            h.raw_mr_cls(foreign, api=h.api())  # type: ignore[call-arg]


class TestContracts:
    def test_zero_byte_allocations_get_unique_freeable_pointers(self, h: Harness) -> None:
        mr, api = _mr(h, False)
        stream = _stream(h, api)
        first = mr.allocate(0, stream)
        second = mr.allocate(0, stream)
        assert first != 0
        assert second != 0
        assert first != second
        # The driver never sees a zero-byte request (a zero-byte malloc
        # returns NULL, which could not be tracked or freed).
        assert (h.platform.symbols.malloc, 1) in api.calls
        mr.deallocate(first, 0, stream)
        mr.deallocate(second, 0, stream)
        assert mr._debug_live_count() == 0

    def test_guaranteed_alignment_is_256(self, h: Harness) -> None:
        mr, _ = _mr(h, False)
        assert mr.guaranteed_alignment() == 256

    def test_live_count_tracks_outstanding_allocations(self, h: Harness) -> None:
        mr, api = _mr(h, False)
        stream = _stream(h, api)
        ptrs = [mr.allocate(nbytes, stream) for nbytes in (16, 32, 64)]
        assert mr._debug_live_count() == 3
        for ptr, nbytes in zip(ptrs, (16, 32, 64), strict=True):
            mr.deallocate(ptr, nbytes, stream)
        assert mr._debug_live_count() == 0

    def test_repr_names_the_device_and_mode(self, h: Harness) -> None:
        mr, _ = _mr(h, True)
        assert h.name in repr(mr)
        assert type(mr).__name__ in repr(mr)


@pytest.fixture
def _fake_rmm_stream(h: Harness, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(h.mrs_module, "_rmm_stream_class", lambda: FakeRmmStream)


def _wrapper(h: Harness, inner: Any) -> RmmLikeMemoryResource:
    return cast(RmmLikeMemoryResource, h.rmm_mr_cls(inner, h.device0))  # type: ignore[call-arg]


@pytest.mark.usefixtures("_fake_rmm_stream")
class TestRmmWrappers:
    def test_forwards_allocate_with_a_translated_stream(self, h: Harness) -> None:
        inner = FakeRmmMemoryResource()
        mr = _wrapper(h, inner)
        stream = h.stream_cls(h.device0, 0xABC, h.api())
        mr.allocate(64, stream)
        name, nbytes, translated = inner.calls[0]
        assert (name, nbytes) == ("allocate", 64)
        assert isinstance(translated, FakeRmmStream)
        # Translation rides the CUDA stream protocol — hipMM's port keeps
        # it verbatim: its Stream reads the wrapped object's
        # __cuda_stream__ too (design §5.2, §5.3).
        assert translated.wrapped is stream
        assert stream.__cuda_stream__() == (0, 0xABC)

    def test_forwards_deallocate_with_a_translated_stream(self, h: Harness) -> None:
        inner = FakeRmmMemoryResource()
        mr = _wrapper(h, inner)
        stream = h.stream_cls(h.device0, 0xABC, h.api())
        ptr = mr.allocate(64, stream)
        mr.deallocate(ptr, 64, stream)
        name, freed_ptr, nbytes, translated = inner.calls[1]
        assert (name, freed_ptr, nbytes) == ("deallocate", ptr, 64)
        assert isinstance(translated, FakeRmmStream)
        assert translated.wrapped is stream

    def test_zero_byte_requests_are_bumped_to_one_byte_both_ways(self, h: Harness) -> None:
        inner = FakeRmmMemoryResource()
        mr = _wrapper(h, inner)
        stream = h.stream_cls(h.device0, 0, h.api())
        first = mr.allocate(0, stream)
        second = mr.allocate(0, stream)
        assert first != second
        mr.deallocate(first, 0, stream)
        assert inner.calls[0][1] == 1
        assert inner.calls[2][2] == 1

    def test_holds_the_wrapped_mr_strongly(self, h: Harness) -> None:
        inner = FakeRmmMemoryResource()
        ref = weakref.ref(inner)
        mr = _wrapper(h, inner)
        del inner
        gc.collect()
        assert ref() is not None
        assert mr.inner is ref()

    def test_capability_probes(self, h: Harness) -> None:
        mr = _wrapper(h, FakeRmmMemoryResource())
        assert mr.stream_ordered is True
        assert mr.guaranteed_alignment() == 256
        assert mr.device == h.device0

    def test_cpu_devices_are_rejected(self, h: Harness) -> None:
        with pytest.raises(ValueError, match="cpu"):
            h.rmm_mr_cls(FakeRmmMemoryResource(), _CPU)  # type: ignore[call-arg]

    def test_sibling_platform_devices_are_rejected(self, h: Harness) -> None:
        foreign = Device(h.foreign_device_type, 0)
        with pytest.raises(ValueError, match=h.name):
            h.rmm_mr_cls(FakeRmmMemoryResource(), foreign)  # type: ignore[call-arg]

    def test_double_free_raises_without_forwarding(self, h: Harness) -> None:
        inner = FakeRmmMemoryResource()
        mr = _wrapper(h, inner)
        stream = h.stream_cls(h.device0, 0, h.api())
        ptr = mr.allocate(64, stream)
        mr.deallocate(ptr, 64, stream)
        calls_before = list(inner.calls)
        with pytest.raises(ValueError, match="live allocation"):
            mr.deallocate(ptr, 64, stream)
        assert inner.calls == calls_before

    def test_size_mismatched_free_raises_and_keeps_the_allocation_live(self, h: Harness) -> None:
        inner = FakeRmmMemoryResource()
        mr = _wrapper(h, inner)
        stream = h.stream_cls(h.device0, 0, h.api())
        ptr = mr.allocate(64, stream)
        with pytest.raises(ValueError, match="size mismatch"):
            mr.deallocate(ptr, 32, stream)
        assert mr._debug_live_count() == 1

    def test_inner_allocation_failures_propagate_untouched(self, h: Harness) -> None:
        marker = MemoryError("pool exhausted")

        class _ExplodingInner:
            def allocate(self, nbytes: int, stream: Any) -> int:
                raise marker

            def deallocate(self, ptr: int, nbytes: int, stream: Any) -> None:
                raise AssertionError("never reached")

        mr = _wrapper(h, _ExplodingInner())
        stream = h.stream_cls(h.device0, 0, h.api())
        with pytest.raises(MemoryError) as excinfo:
            mr.allocate(64, stream)
        assert excinfo.value is marker
        assert mr._debug_live_count() == 0


@pytest.mark.skipif(importlib.util.find_spec("rmm") is not None, reason="rmm is installed here")
def test_stream_translation_without_rmm_raises_runtime_unavailable() -> None:
    with pytest.raises(RuntimeUnavailableError, match="devmm\\[cuda\\]"):
        mrs_cuda._rmm_stream_class()
    with pytest.raises(RuntimeUnavailableError, match="devmm\\[rocm\\]"):
        mrs_rocm._rmm_stream_class()


class TestSharedShim:
    """The §4.2 payoff pinned explicitly: both platforms' MRs are the same
    classes over different symbol/error tables."""

    def test_raw_mrs_share_the_gpulib_base(self) -> None:
        for harness in HARNESSES.values():
            assert issubclass(harness.raw_mr_cls, GpuRuntimeMemoryResource)
            assert issubclass(harness.rmm_mr_cls, RmmLikeMemoryResource)
