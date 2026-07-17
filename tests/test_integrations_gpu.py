"""CUDA hardware (T2) integrations suite (design §5.2, §5.4, §6, §9):
`CupyAllocatorMemoryResource` consumption observed through CuPy pool
introspection (`used_bytes()` deltas, per-stream arena reuse, current-device
context), the devmm -> CuPy install round trip, and the Numba EMM plugin —
Numba's own memory test protocol run under the plugin plus a `@cuda.jit`
kernel writing into devmm-accounted memory.

Every test carries `gpu_cuda`: opt in with DEVMM_GPU=cuda on a CUDA machine.
"""

from __future__ import annotations

import gc
import os
import subprocess
import sys
from typing import Any

import pytest

from devmm import Device, StatisticsAdaptor, using_memory_resource
from devmm._core.stream import Stream
from devmm._runtimes.cuda import CudaRuntime
from devmm.integrations import cupy as integrations_cupy
from devmm.integrations import numba as integrations_numba
from devmm.mrs.cuda import CudaRuntimeMemoryResource, CupyAllocatorMemoryResource

pytestmark = pytest.mark.gpu_cuda

_DEVICE = Device.from_string("cuda:0")


@pytest.fixture(scope="module")
def runtime() -> CudaRuntime:
    return CudaRuntime()


@pytest.fixture
def stream(runtime: CudaRuntime) -> Stream:
    return runtime.create_stream(_DEVICE)


class TestCupyAllocatorMr:
    def test_pool_used_bytes_delta_and_return_to_pool(self, stream: Stream) -> None:
        cp = pytest.importorskip("cupy")
        pool = cp.cuda.MemoryPool()
        mr = CupyAllocatorMemoryResource(pool.malloc, _DEVICE)
        before = pool.used_bytes()
        ptr = mr.allocate(4096, stream)
        assert pool.used_bytes() >= before + 4096
        mr.deallocate(ptr, 4096, stream)
        # The block goes back to the pool (used drops), not to the driver
        # (the pool keeps it cached).
        assert pool.used_bytes() == before
        assert pool.total_bytes() >= 4096

    def test_default_allocator_draws_from_cupys_current_pool(self, stream: Stream) -> None:
        cp = pytest.importorskip("cupy")
        pool = cp.get_default_memory_pool()
        mr = CupyAllocatorMemoryResource(device=_DEVICE)
        before = pool.used_bytes()
        ptr = mr.allocate(8192, stream)
        assert pool.used_bytes() >= before + 8192
        mr.deallocate(ptr, 8192, stream)
        assert pool.used_bytes() == before

    def test_allocation_is_keyed_to_the_callers_stream(self, runtime: CudaRuntime) -> None:
        # CuPy pools cache freed blocks per allocation stream: freeing and
        # re-allocating the same size on the same stream must reuse the
        # cached block, proving the ExternalStream context was current.
        cp = pytest.importorskip("cupy")
        pool = cp.cuda.MemoryPool()
        mr = CupyAllocatorMemoryResource(pool.malloc, _DEVICE)
        stream_a = runtime.create_stream(_DEVICE)
        ptr = mr.allocate(2048, stream_a)
        mr.deallocate(ptr, 2048, stream_a)
        again = mr.allocate(2048, stream_a)
        assert again == ptr
        mr.deallocate(again, 2048, stream_a)


class TestCupyInstall:
    def test_cupy_arrays_allocate_through_the_installed_mr(self) -> None:
        cp = pytest.importorskip("cupy")
        stats = StatisticsAdaptor(CudaRuntimeMemoryResource(_DEVICE))
        with integrations_cupy.install(stats):
            array = cp.arange(1024, dtype=cp.float32)
            assert stats.current_bytes >= 4096
            assert float(array.sum()) == float(1023 * 1024 / 2)
        # Uninstalled, but the array still owns its devmm buffer.
        assert stats.current_bytes >= 4096
        del array
        gc.collect()
        assert stats.current_bytes == 0


class TestNumbaEmm:
    def test_kernel_writes_into_devmm_accounted_memory(self) -> None:
        numba_cuda = pytest.importorskip("numba.cuda")
        np = pytest.importorskip("numpy")
        stats = StatisticsAdaptor(CudaRuntimeMemoryResource(_DEVICE))
        handle = integrations_numba.install()
        try:
            with using_memory_resource(stats):
                array = numba_cuda.to_device(np.zeros(256, dtype=np.float32))
                assert stats.current_bytes >= 1024

                def fill_kernel(data: Any) -> None:
                    i = numba_cuda.grid(1)
                    if i < data.size:
                        data[i] = 3.0

                fill = numba_cuda.jit(fill_kernel)
                fill[1, 256](array)
                numba_cuda.synchronize()
                host = array.copy_to_host()
                assert (host == 3.0).all()
        finally:
            handle.uninstall()

    def test_plugin_passes_numbas_own_memory_test_protocol(self) -> None:
        pytest.importorskip("numba")
        env = dict(os.environ, NUMBA_CUDA_MEMORY_MANAGER="devmm.integrations.numba")
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "numba.runtests",
                "numba.cuda.tests.cudadrv.test_cuda_memory",
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
        )
        assert result.returncode == 0, result.stderr
