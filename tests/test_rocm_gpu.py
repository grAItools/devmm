"""ROCm hardware (T3) suite: the MR conformance contract over
`HipRuntimeMemoryResource` (sync and async) and `HipmmMemoryResource` with
writes/reads through the runtime's `memcpy`, and DLPack round-trips through
CuPy-ROCm and PyTorch-ROCm as available on the runner (design §5.3, §9).

Every test carries `gpu_rocm`: opt in with DEVMM_GPU=rocm on a ROCm machine.
"""

from __future__ import annotations

import ctypes

import pytest

from devmm import Aligned, Device, DeviceMemoryResource, RowMajor, empty
from devmm._core.stream import Stream
from devmm._runtimes._gpulib import rmm_module_platform
from devmm._runtimes.base import CopyKind
from devmm._runtimes.rocm import HipRuntime
from devmm.mrs.rocm import HipmmMemoryResource, HipRuntimeMemoryResource

pytestmark = pytest.mark.gpu_rocm

_DEVICE = Device.from_string("rocm:0")


@pytest.fixture(scope="module")
def runtime() -> HipRuntime:
    return HipRuntime()


@pytest.fixture
def stream(runtime: HipRuntime) -> Stream:
    return runtime.create_stream(_DEVICE)


@pytest.fixture(params=["runtime-sync", "runtime-async", "hipmm"])
def gpu_mr(request: pytest.FixtureRequest, runtime: HipRuntime) -> DeviceMemoryResource:
    if request.param == "runtime-sync":
        return HipRuntimeMemoryResource(_DEVICE, async_alloc=False, api=runtime.api)
    if request.param == "runtime-async":
        mr = HipRuntimeMemoryResource(_DEVICE, async_alloc="auto", api=runtime.api)
        if not mr.stream_ordered:
            pytest.skip("driver does not support hipMallocAsync")
        return mr
    rmm = pytest.importorskip("rmm")
    if rmm_module_platform(rmm) != "rocm":
        pytest.skip("the installed rmm module does not target ROCm (design §4.2)")
    return HipmmMemoryResource(rmm.mr.HipMemoryResource(), _DEVICE)


def _pattern(nbytes: int) -> bytes:
    # Period 251 (prime) so the pattern can never line up with power-of-two
    # allocation granularities and mask an addressing bug.
    return bytes(i % 251 for i in range(nbytes))


def _write(runtime: HipRuntime, stream: Stream, ptr: int, data: bytes) -> None:
    staged = (ctypes.c_char * len(data)).from_buffer_copy(data)
    runtime.memcpy(ptr, ctypes.addressof(staged), len(data), CopyKind.HOST_TO_DEVICE, stream)


def _read(runtime: HipRuntime, stream: Stream, ptr: int, nbytes: int) -> bytes:
    staged = (ctypes.c_char * nbytes)()
    runtime.memcpy(ctypes.addressof(staged), ptr, nbytes, CopyKind.DEVICE_TO_HOST, stream)
    return staged.raw


class TestGpuMrConformance:
    """The phase-4 conformance contract, restated for device memory: raw
    pointers are exercised through the runtime's `memcpy` instead of host
    `ctypes` dereferences (design §9)."""

    def test_write_then_read_is_byte_exact(
        self, runtime: HipRuntime, gpu_mr: DeviceMemoryResource, stream: Stream
    ) -> None:
        data = _pattern(1024)
        ptr = gpu_mr.allocate(len(data), stream)
        _write(runtime, stream, ptr, data)
        assert _read(runtime, stream, ptr, len(data)) == data
        gpu_mr.deallocate(ptr, len(data), stream)

    def test_live_allocations_do_not_alias(
        self, runtime: HipRuntime, gpu_mr: DeviceMemoryResource, stream: Stream
    ) -> None:
        first = gpu_mr.allocate(256, stream)
        _write(runtime, stream, first, b"\xaa" * 256)
        second = gpu_mr.allocate(256, stream)
        _write(runtime, stream, second, b"\x55" * 256)
        assert _read(runtime, stream, first, 256) == b"\xaa" * 256
        gpu_mr.deallocate(first, 256, stream)
        assert _read(runtime, stream, second, 256) == b"\x55" * 256
        gpu_mr.deallocate(second, 256, stream)

    def test_guaranteed_alignment_is_honest(
        self, gpu_mr: DeviceMemoryResource, stream: Stream
    ) -> None:
        alignment = gpu_mr.guaranteed_alignment()
        assert alignment == 256
        for nbytes in (1, 3, 17, 255, 4096):
            ptr = gpu_mr.allocate(nbytes, stream)
            assert ptr % alignment == 0
            gpu_mr.deallocate(ptr, nbytes, stream)

    def test_bookkeeping_is_empty_after_alloc_free_pairs(
        self, gpu_mr: DeviceMemoryResource, stream: Stream
    ) -> None:
        live = [(gpu_mr.allocate(nbytes, stream), nbytes) for nbytes in range(0, 64, 7)]
        assert gpu_mr._debug_live_count() == len(live)  # type: ignore[attr-defined]
        for ptr, nbytes in live:
            gpu_mr.deallocate(ptr, nbytes, stream)
        assert gpu_mr._debug_live_count() == 0  # type: ignore[attr-defined]

    def test_double_free_raises(self, gpu_mr: DeviceMemoryResource, stream: Stream) -> None:
        ptr = gpu_mr.allocate(64, stream)
        gpu_mr.deallocate(ptr, 64, stream)
        with pytest.raises(ValueError):
            gpu_mr.deallocate(ptr, 64, stream)

    def test_freeing_an_unknown_pointer_raises(
        self, gpu_mr: DeviceMemoryResource, stream: Stream
    ) -> None:
        ptr = gpu_mr.allocate(64, stream)
        with pytest.raises(ValueError):
            gpu_mr.deallocate(ptr + 1, 64, stream)
        gpu_mr.deallocate(ptr, 64, stream)

    def test_zero_byte_allocation_round_trips(
        self, gpu_mr: DeviceMemoryResource, stream: Stream
    ) -> None:
        first = gpu_mr.allocate(0, stream)
        second = gpu_mr.allocate(0, stream)
        assert first != 0
        assert second != 0
        assert first != second
        gpu_mr.deallocate(first, 0, stream)
        gpu_mr.deallocate(second, 0, stream)

    def test_negative_allocation_size_is_rejected(
        self, gpu_mr: DeviceMemoryResource, stream: Stream
    ) -> None:
        with pytest.raises(ValueError):
            gpu_mr.allocate(-1, stream)


class TestDlpackRoundTrips:
    def test_cupy_round_trip_is_zero_copy(
        self, runtime: HipRuntime, gpu_mr: DeviceMemoryResource, stream: Stream
    ) -> None:
        cp = pytest.importorskip("cupy")
        np = pytest.importorskip("numpy")
        t = empty((3, 5), "float32", device=_DEVICE, mr=gpu_mr, stream=stream, layout=RowMajor())
        expected = np.arange(15, dtype=np.float32).reshape(3, 5)
        _write(runtime, stream, t.buffer.ptr, expected.tobytes())
        arr = cp.from_dlpack(t)
        assert arr.shape == (3, 5)
        np.testing.assert_array_equal(cp.asnumpy(arr), expected)
        arr[0, 0] = 42.0
        cp.cuda.get_current_stream().synchronize()
        raw = _read(runtime, stream, t.buffer.ptr, 4)
        assert np.frombuffer(raw, dtype=np.float32)[0] == 42.0

    def test_cupy_round_trip_with_padded_strides(
        self, runtime: HipRuntime, gpu_mr: DeviceMemoryResource, stream: Stream
    ) -> None:
        cp = pytest.importorskip("cupy")
        np = pytest.importorskip("numpy")
        layout = Aligned(RowMajor(), unit_stride_alignment=128, base_alignment=256)
        t = empty((4, 3), "float32", device=_DEVICE, mr=gpu_mr, stream=stream, layout=layout)
        assert t.layout.strides == (32, 1)
        expected = np.arange(12, dtype=np.float32).reshape(4, 3)
        host = np.zeros((4, 32), dtype=np.float32)
        host[:, :3] = expected
        _write(runtime, stream, t.buffer.ptr, host.tobytes())
        arr = cp.from_dlpack(t)
        assert arr.strides == (128, 4)
        np.testing.assert_array_equal(cp.asnumpy(arr), expected)

    def test_torch_round_trip_with_padded_strides(
        self, runtime: HipRuntime, gpu_mr: DeviceMemoryResource, stream: Stream
    ) -> None:
        torch = pytest.importorskip("torch")
        np = pytest.importorskip("numpy")
        # PyTorch-ROCm reports availability through the `cuda` namespace
        # (`torch.version.hip` names the actual platform).
        if not torch.cuda.is_available() or torch.version.hip is None:
            pytest.skip("torch built without ROCm")
        layout = Aligned(RowMajor(), unit_stride_alignment=128, base_alignment=256)
        t = empty((4, 3), "float32", device=_DEVICE, mr=gpu_mr, stream=stream, layout=layout)
        expected = np.arange(12, dtype=np.float32).reshape(4, 3)
        host = np.zeros((4, 32), dtype=np.float32)
        host[:, :3] = expected
        _write(runtime, stream, t.buffer.ptr, host.tobytes())
        x = torch.from_dlpack(t)
        assert tuple(x.shape) == (4, 3)
        assert tuple(x.stride()) == (32, 1)
        np.testing.assert_array_equal(x.cpu().numpy(), expected)
        x[0, 0] = 42.0
        torch.cuda.synchronize()
        raw = _read(runtime, stream, t.buffer.ptr, 4)
        assert np.frombuffer(raw, dtype=np.float32)[0] == 42.0
