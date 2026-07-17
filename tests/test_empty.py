"""`Tensor` introspection plus the `empty()`/`empty_like()` factories
(design §3.8): registry-default MR resolution, alignment-aware
over-allocation, duck-typed `empty_like`, and `array_api_strict.from_dlpack`
consuming devmm tensors (conformance oracle #2, design §9).
"""

from __future__ import annotations

import numpy as np
import pytest

from devmm import (
    Device,
    DeviceBuffer,
    RowMajor,
    Tensor,
    empty,
    empty_like,
    using_memory_resource,
)
from devmm._core import dtypes
from devmm._core.stream import CpuStream
from devmm._runtimes.base import RuntimeUnavailableError
from devmm.mrs.cpu import MallocMemoryResource
from devmm.testing import RecordingMemoryResource
from tests._dlpack_utils import write_pattern

_CPU = Device.from_string("cpu")


class TestTensor:
    def test_introspection_properties(self) -> None:
        t = empty((2, 3), "float32", mr=MallocMemoryResource())
        assert t.device == _CPU
        assert t.shape == (2, 3)
        assert t.dtype == dtypes.float32
        assert t.strides == t.layout.strides
        assert t.offset == 0
        assert t.read_only is False
        assert t.__dlpack_device__() == (1, 0)

    def test_rejects_a_negative_offset(self) -> None:
        t = empty((2, 3), "float32", mr=MallocMemoryResource())
        with pytest.raises(ValueError):
            Tensor(t.buffer, t.dtype, t.shape, t.layout, offset=-1)

    def test_rejects_a_buffer_too_small_for_the_layout(self) -> None:
        mr = MallocMemoryResource()
        layout = RowMajor()((4, 4), dtypes.float32, _CPU)
        small = DeviceBuffer(16, mr=mr, stream=CpuStream())
        with pytest.raises(ValueError):
            Tensor(small, dtypes.float32, (4, 4), layout)

    def test_rejects_a_rank_mismatched_layout(self) -> None:
        t = empty((2, 3), "float32", mr=MallocMemoryResource())
        with pytest.raises(ValueError):
            Tensor(t.buffer, t.dtype, (2, 3, 4), t.layout)


class TestEmpty:
    def test_uses_the_registry_current_mr_by_default(self) -> None:
        mr = RecordingMemoryResource()
        with using_memory_resource(mr):
            t = empty((2, 2), "float32")
        assert t.buffer.mr is mr

    def test_rejects_an_mr_on_another_device(self) -> None:
        mr = RecordingMemoryResource(device=Device.from_string("cuda:0"))
        with pytest.raises(ValueError):
            empty((2,), "float32", mr=mr)

    def test_non_cpu_default_streams_require_a_runtime(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The default stream resolves through `runtime_for(device)`; forcing
        # the cpu runtime makes "no runtime serves cuda" deterministic even
        # on hosts where the real CUDA runtime would load.
        monkeypatch.setenv("DEVMM_RUNTIME", "cpu")
        device = Device.from_string("cuda:0")
        mr = RecordingMemoryResource(device=device)
        with pytest.raises(RuntimeUnavailableError):
            empty((2,), "float32", device=device, mr=mr)

    def test_accepts_a_concrete_layout(self) -> None:
        layout = RowMajor()((2, 3), dtypes.float32, _CPU)
        t = empty((2, 3), "float32", layout=layout, mr=MallocMemoryResource())
        assert t.layout is layout

    def test_validates_a_concrete_layout_against_the_shape(self) -> None:
        layout = RowMajor()((2, 3), dtypes.float32, _CPU)
        with pytest.raises(ValueError):
            empty((4, 5), "float32", layout=layout, mr=MallocMemoryResource())

    def test_over_allocates_when_the_mr_guarantee_is_weaker(self) -> None:
        # DeviceOptimal on CPU wants a 64-byte base; an 8-byte-aligned MR
        # forces one extra alignment span and an element offset to the next
        # boundary (design §3.6).
        mr = RecordingMemoryResource(guaranteed_alignment=8)
        t = empty((4, 4), "float32", mr=mr)
        assert t.layout.base_alignment == 64
        assert t.buffer.nbytes == t.layout.required_nbytes + 64
        assert (t.buffer.ptr + t.offset * 4) % 64 == 0

    def test_over_alignment_is_best_effort_on_a_weakly_aligned_pointer(self) -> None:
        # A guaranteed_alignment=1 MR can return a pointer that is not even
        # itemsize-aligned; element offsets cannot reach the base boundary
        # then, so the tensor stays at the buffer start (still over-allocated,
        # aligned only as well as the MR delivered).
        mr = RecordingMemoryResource(guaranteed_alignment=1)
        t = empty((4, 4), "float32", mr=mr)
        assert t.buffer.ptr % 4 != 0
        assert t.buffer.nbytes == t.layout.required_nbytes + 64
        assert t.offset == 0

    def test_skips_over_allocation_when_the_mr_guarantee_suffices(self) -> None:
        mr = RecordingMemoryResource(guaranteed_alignment=256)
        t = empty((4, 4), "float32", mr=mr)
        assert t.buffer.nbytes == t.layout.required_nbytes
        assert t.offset == 0

    def test_zero_size_shapes_allocate_no_bytes(self) -> None:
        t = empty((0, 5), "float32", mr=MallocMemoryResource())
        assert t.layout.required_nbytes == 0
        assert t.buffer.nbytes == 0
        assert t.offset == 0


class _ShapeDtypeOnly:
    """Duck-typed array with no `__dlpack_device__`."""

    shape = (2, 2)
    dtype = np.dtype("float32")


class TestEmptyLike:
    def test_duck_types_numpy_arrays(self) -> None:
        src = np.zeros((3, 4), dtype=np.int16)
        t = empty_like(src, mr=MallocMemoryResource())
        assert t.shape == (3, 4)
        assert t.dtype == dtypes.int16
        assert t.device == _CPU

    def test_duck_types_array_api_strict_arrays(self) -> None:
        xp = pytest.importorskip("array_api_strict")
        src = xp.ones((2, 5), dtype=xp.float64)
        t = empty_like(src, mr=MallocMemoryResource())
        assert t.shape == (2, 5)
        assert t.dtype == dtypes.float64
        assert t.device == _CPU

    def test_duck_types_devmm_tensors(self) -> None:
        src = empty((4, 2), "complex64", mr=MallocMemoryResource())
        t = empty_like(src, mr=MallocMemoryResource())
        assert t.shape == (4, 2)
        assert t.dtype == dtypes.complex64
        assert t.device == _CPU

    def test_dtype_and_layout_overrides_win(self) -> None:
        src = np.zeros((3, 4), dtype=np.int16)
        t = empty_like(src, dtype="float64", layout=RowMajor(), mr=MallocMemoryResource())
        assert t.dtype == dtypes.float64
        assert t.layout.policy == RowMajor()

    def test_requires_dlpack_device_or_an_explicit_device(self) -> None:
        with pytest.raises(TypeError):
            empty_like(_ShapeDtypeOnly(), mr=MallocMemoryResource())
        t = empty_like(_ShapeDtypeOnly(), device=_CPU, mr=MallocMemoryResource())
        assert t.device == _CPU
        assert t.shape == (2, 2)


def test_array_api_strict_consumes_devmm_tensors() -> None:
    xp = pytest.importorskip("array_api_strict")
    t = empty((3, 4), "float32", mr=MallocMemoryResource())
    expected = write_pattern(t, np.dtype("float32"))
    consumed = xp.from_dlpack(t)
    assert consumed.shape == (3, 4)
    assert consumed.dtype == xp.float32
    np.testing.assert_array_equal(np.from_dlpack(consumed), expected)
