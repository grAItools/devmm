"""CPU memory-resource tests: both MRs run the reusable conformance suite
(`devmm.testing.mr_conformance`), plus the MR-specific contracts —
`BytearrayMemoryResource` pinning/release, `MallocMemoryResource` exact
alignment + allocator-family tracking, and the Linux RSS leak canary
(design §5.1, §9).
"""

from __future__ import annotations

import gc
import subprocess
import sys
import textwrap
import weakref

import pytest

from devmm import Device
from devmm._core.stream import CpuStream
from devmm.mrs.cpu import BytearrayMemoryResource, MallocMemoryResource
from devmm.testing._mr_conformance import MemoryResourceConformance

STREAM = CpuStream()

CpuMR = BytearrayMemoryResource | MallocMemoryResource

MR_TYPES: tuple[type[CpuMR], ...] = (BytearrayMemoryResource, MallocMemoryResource)
MR_IDS = ("bytearray", "malloc")


class TestBytearrayConformance(MemoryResourceConformance):
    def make_mr(self, *, alignment: int | None = None) -> BytearrayMemoryResource:
        if alignment is None:
            return BytearrayMemoryResource()
        return BytearrayMemoryResource(alignment=alignment)


class TestMallocConformance(MemoryResourceConformance):
    def make_mr(self, *, alignment: int | None = None) -> MallocMemoryResource:
        if alignment is None:
            return MallocMemoryResource()
        return MallocMemoryResource(alignment=alignment)


@pytest.mark.parametrize("mr_type", MR_TYPES, ids=MR_IDS)
def test_cpu_mrs_reject_non_cpu_devices(mr_type: type[CpuMR]) -> None:
    with pytest.raises(ValueError):
        mr_type(Device.from_string("cuda:0"))


@pytest.mark.parametrize("alignment", (0, 3, -8))
@pytest.mark.parametrize("mr_type", MR_TYPES, ids=MR_IDS)
def test_cpu_mrs_reject_non_power_of_two_alignment(mr_type: type[CpuMR], alignment: int) -> None:
    with pytest.raises(ValueError):
        mr_type(alignment=alignment)


@pytest.mark.parametrize("mr_type", MR_TYPES, ids=MR_IDS)
def test_cpu_mrs_are_not_stream_ordered(mr_type: type[CpuMR]) -> None:
    assert mr_type().stream_ordered is False


@pytest.mark.parametrize("mr_type", MR_TYPES, ids=MR_IDS)
def test_allocation_failure_raises_memory_error_with_context(mr_type: type[CpuMR]) -> None:
    mr = mr_type()
    # 2**61 bytes exceeds any 64-bit machine's address space, so the native
    # allocation must fail deterministically.
    with pytest.raises(MemoryError, match="cpu:0"):
        mr.allocate(1 << 61, STREAM)
    assert mr._debug_live_count() == 0


def test_bytearray_guaranteed_alignment_is_one_even_when_configured() -> None:
    assert BytearrayMemoryResource().guaranteed_alignment() == 1
    assert BytearrayMemoryResource(alignment=64).guaranteed_alignment() == 1


def test_bytearray_pins_backing_store_while_allocated() -> None:
    mr = BytearrayMemoryResource()
    ptr = mr.allocate(16, STREAM)
    # Hold only the backing store: holding the pin too would keep the buffer
    # export alive past deallocate and mask the unpinning.
    backing = mr._live[ptr][0]
    with pytest.raises(BufferError):
        backing.append(0)
    mr.deallocate(ptr, 16, STREAM)
    # Unpinned after free: the resize that just raised now succeeds.
    backing.append(0)
    assert len(backing) == 17


def test_bytearray_backing_store_dies_after_free() -> None:
    mr = BytearrayMemoryResource()
    ptr = mr.allocate(16, STREAM)
    probe = weakref.ref(mr._live[ptr][0])
    assert probe() is not None
    mr.deallocate(ptr, 16, STREAM)
    gc.collect()
    assert probe() is None


def test_malloc_guaranteed_alignment_reports_the_configured_alignment() -> None:
    assert MallocMemoryResource().guaranteed_alignment() == 64
    assert MallocMemoryResource(alignment=256).guaranteed_alignment() == 256


def test_malloc_records_the_platform_family() -> None:
    mr = MallocMemoryResource()
    ptr = mr.allocate(32, STREAM)
    _nbytes, family = mr._live[ptr]
    assert family.name == ("windows" if sys.platform == "win32" else "posix")
    mr.deallocate(ptr, 32, STREAM)


def test_malloc_size_mismatched_free_raises_and_keeps_the_allocation_live() -> None:
    mr = MallocMemoryResource()
    ptr = mr.allocate(32, STREAM)
    with pytest.raises(ValueError, match="size mismatch"):
        mr.deallocate(ptr, 16, STREAM)
    assert mr._debug_live_count() == 1
    mr.deallocate(ptr, 32, STREAM)
    assert mr._debug_live_count() == 0


def test_malloc_deallocate_frees_through_the_recorded_family() -> None:
    """Deallocation dispatches per pointer, not per platform: on Windows an
    `_aligned_malloc` pointer must go back through `_aligned_free`."""
    mr = MallocMemoryResource()
    ptr = mr.allocate(32, STREAM)
    nbytes, real_family = mr._live[ptr]
    freed: list[int] = []

    class _StubFamily:
        name = "stub"

        def alloc(self, nbytes: int, alignment: int) -> int:
            raise AssertionError("deallocate must never allocate")

        def free(self, ptr: int) -> None:
            freed.append(ptr)

    mr._live[ptr] = (nbytes, _StubFamily())
    mr.deallocate(ptr, 32, STREAM)
    assert freed == [ptr]
    # Release the real memory the stub intercepted.
    real_family.free(ptr)


_LEAK_CANARY = textwrap.dedent(
    """
    from devmm._core.stream import CpuStream
    from devmm.mrs.cpu import MallocMemoryResource

    def rss_kib() -> int:
        with open("/proc/self/status") as status:
            for line in status:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
        raise SystemExit("VmRSS not found in /proc/self/status")

    mr = MallocMemoryResource()
    stream = CpuStream()
    nbytes = 1 << 20
    for _ in range(1_000):
        mr.deallocate(mr.allocate(nbytes, stream), nbytes, stream)
    before = rss_kib()
    for _ in range(100_000):
        mr.deallocate(mr.allocate(nbytes, stream), nbytes, stream)
    print(rss_kib() - before)
    """
)


@pytest.mark.slow
@pytest.mark.skipif(sys.platform != "linux", reason="reads VmRSS from /proc/self/status")
def test_malloc_leak_canary_rss_growth_is_bounded() -> None:
    result = subprocess.run(
        [sys.executable, "-c", _LEAK_CANARY], capture_output=True, text=True, check=True
    )
    growth_kib = int(result.stdout.strip())
    # 10**5 leaked MiB would be ~100 GiB; anything under 32 MiB is allocator
    # noise, not a leak.
    assert growth_kib < 32 * 1024
