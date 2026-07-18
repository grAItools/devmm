"""CUDA hardware (T2) suite: the MR conformance contract over
`CudaRuntimeMemoryResource` (sync and async) and `RmmMemoryResource` with
writes/reads through the runtime's `memcpy`, DLPack round-trips through CuPy
and PyTorch (padded strides included), the stream-race canary, and rmm pool
statistics vs `StatisticsAdaptor` (design §5.2, §9).

Every test carries `gpu_cuda`: opt in with DEVMM_GPU=cuda on a CUDA machine.
"""

from __future__ import annotations

import ctypes
from typing import Any

import pytest

from devmm import Aligned, Device, DeviceMemoryResource, RowMajor, StatisticsAdaptor, empty
from devmm._core.stream import Stream
from devmm._runtimes.base import CopyKind
from devmm._runtimes.cuda import CudaRuntime
from devmm.mrs.cuda import CudaRuntimeMemoryResource, RmmMemoryResource

pytestmark = pytest.mark.gpu_cuda

_DEVICE = Device.from_string("cuda:0")


@pytest.fixture(scope="module")
def runtime() -> CudaRuntime:
    return CudaRuntime()


@pytest.fixture
def stream(runtime: CudaRuntime) -> Stream:
    return runtime.create_stream(_DEVICE)


@pytest.fixture(params=["runtime-sync", "runtime-async", "rmm"])
def gpu_mr(request: pytest.FixtureRequest, runtime: CudaRuntime) -> DeviceMemoryResource:
    if request.param == "runtime-sync":
        return CudaRuntimeMemoryResource(_DEVICE, async_alloc=False, api=runtime.api)
    if request.param == "runtime-async":
        mr = CudaRuntimeMemoryResource(_DEVICE, async_alloc="auto", api=runtime.api)
        if not mr.stream_ordered:
            pytest.skip("driver does not support cudaMallocAsync")
        return mr
    rmm = pytest.importorskip("rmm")
    return RmmMemoryResource(rmm.mr.CudaMemoryResource(), _DEVICE)


def _pattern(nbytes: int) -> bytes:
    # Period 251 (prime) so the pattern can never line up with power-of-two
    # allocation granularities and mask an addressing bug.
    return bytes(i % 251 for i in range(nbytes))


def _write(runtime: CudaRuntime, stream: Stream, ptr: int, data: bytes) -> None:
    staged = (ctypes.c_char * len(data)).from_buffer_copy(data)
    runtime.memcpy(ptr, ctypes.addressof(staged), len(data), CopyKind.HOST_TO_DEVICE, stream)


def _read(runtime: CudaRuntime, stream: Stream, ptr: int, nbytes: int) -> bytes:
    staged = (ctypes.c_char * nbytes)()
    runtime.memcpy(ctypes.addressof(staged), ptr, nbytes, CopyKind.DEVICE_TO_HOST, stream)
    return staged.raw


class TestGpuMrConformance:
    """The phase-4 conformance contract, restated for device memory: raw
    pointers are exercised through the runtime's `memcpy` instead of host
    `ctypes` dereferences (design §9)."""

    def test_write_then_read_is_byte_exact(
        self, runtime: CudaRuntime, gpu_mr: DeviceMemoryResource, stream: Stream
    ) -> None:
        data = _pattern(1024)
        ptr = gpu_mr.allocate(len(data), stream)
        _write(runtime, stream, ptr, data)
        assert _read(runtime, stream, ptr, len(data)) == data
        gpu_mr.deallocate(ptr, len(data), stream)

    def test_live_allocations_do_not_alias(
        self, runtime: CudaRuntime, gpu_mr: DeviceMemoryResource, stream: Stream
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
        self, runtime: CudaRuntime, gpu_mr: DeviceMemoryResource, stream: Stream
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
        self, runtime: CudaRuntime, gpu_mr: DeviceMemoryResource, stream: Stream
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
        self, runtime: CudaRuntime, gpu_mr: DeviceMemoryResource, stream: Stream
    ) -> None:
        torch = pytest.importorskip("torch")
        np = pytest.importorskip("numpy")
        if not torch.cuda.is_available():
            pytest.skip("torch built without CUDA")
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


_SLOW_FILL_SRC = r"""
extern "C" __global__
void slow_fill(float* data, unsigned long long n, float value) {
    unsigned long long i = (unsigned long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    // Busy-loop long enough that an unordered consumer can observe stale
    // zeros; the accumulator keeps the loop from being optimized away.
    float acc = 0.0f;
    for (int k = 0; k < 2000; ++k) acc += 1e-30f;
    data[i] = value + acc;
}
"""


def _run_canary(runtime: CudaRuntime, cp: Any) -> bool:
    """Producer fills on stream A; consumer imports via `__dlpack__(stream=B)`
    and sums on B. True iff every element was observed as filled."""
    np = pytest.importorskip("numpy")
    n = 1 << 22
    producer = runtime.create_stream(_DEVICE)
    mr = CudaRuntimeMemoryResource(_DEVICE, api=runtime.api)
    t = empty((n,), "float32", device=_DEVICE, mr=mr, stream=producer)
    kernel = cp.RawKernel(_SLOW_FILL_SRC, "slow_fill")
    mem = cp.cuda.UnownedMemory(t.buffer.ptr, t.buffer.nbytes, t)
    view = cp.ndarray((n,), cp.float32, cp.cuda.MemoryPointer(mem, 0))
    with cp.cuda.ExternalStream(producer.handle):
        view.fill(0.0)
    producer.synchronize()
    blocks = -(-n // 256)
    with cp.cuda.ExternalStream(producer.handle):
        kernel((blocks,), (256,), (view, np.uint64(n), np.float32(1.0)))
    consumer = cp.cuda.Stream(non_blocking=True)
    with consumer:
        arr = cp.from_dlpack(t)
        total = float(arr.sum())
    consumer.synchronize()
    producer.synchronize()
    return total >= float(n)


def test_stream_race_canary_is_ordered_with_the_handoff(runtime: CudaRuntime) -> None:
    cp = pytest.importorskip("cupy")
    assert _run_canary(runtime, cp)


def test_stream_race_canary_can_misorder_without_the_handoff(
    runtime: CudaRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    cp = pytest.importorskip("cupy")
    monkeypatch.setattr(
        CudaRuntime, "make_stream_wait", lambda self, consumer_handle, producer: None
    )
    if _run_canary(runtime, cp):
        # The race is real but not guaranteed to manifest on every run/GPU;
        # a lucky ordering is not a pass for "demonstrably wrong".
        pytest.skip("race did not manifest with the handoff disabled (best-effort)")


_STATISTICS_FIELDS = ("current_bytes", "peak_bytes", "total_bytes")


def _allocation_counts(adaptor: Any) -> dict[str, int]:
    """rmm >= 26.06 reports `allocation_counts` as a `Statistics` object;
    older rmm returned a plain dict. Normalise to a dict either way, failing
    loudly if a field is missing rather than reporting a subset — an upstream
    rename must not read as a pass."""
    counts = adaptor.allocation_counts
    if not isinstance(counts, dict):
        counts = {
            name: getattr(counts, name) for name in _STATISTICS_FIELDS if hasattr(counts, name)
        }
    missing = [name for name in _STATISTICS_FIELDS if name not in counts]
    assert not missing, f"rmm allocation_counts is missing {missing}"
    return counts


def test_rmm_pool_statistics_agree_with_statistics_adaptor(
    runtime: CudaRuntime, stream: Stream
) -> None:
    rmm = pytest.importorskip("rmm")
    upstream = rmm.mr.StatisticsResourceAdaptor(
        rmm.mr.PoolMemoryResource(rmm.mr.CudaMemoryResource(), initial_pool_size=1 << 24)
    )
    mr = StatisticsAdaptor(RmmMemoryResource(upstream, _DEVICE))
    sizes = (256, 1024, 4096)
    ptrs = [mr.allocate(nbytes, stream) for nbytes in sizes]
    counts = _allocation_counts(upstream)
    assert counts["current_bytes"] == mr.current_bytes == sum(sizes)
    assert counts["peak_bytes"] == mr.peak_bytes == sum(sizes)
    for ptr, nbytes in zip(ptrs, sizes, strict=True):
        mr.deallocate(ptr, nbytes, stream)
    counts = _allocation_counts(upstream)
    assert counts["current_bytes"] == mr.current_bytes == 0
    assert counts["total_bytes"] == mr.total_bytes == sum(sizes)
