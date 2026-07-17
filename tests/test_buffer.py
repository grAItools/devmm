"""`DeviceBuffer` contract tests: deallocate-exactly-once lifecycle against
the recording MR, the `weakref.finalize` GC safety net (drop, cycle,
no-resurrect, interpreter shutdown), the `closed` use-after-free guard, and
byte-exact host-copy round-trips on the real CPU MRs (design §3.5).
"""

from __future__ import annotations

import gc
import subprocess
import sys
import textwrap
import weakref

import pytest

from devmm import Device, DeviceBuffer, DeviceType, Stream
from devmm._core.stream import CpuStream
from devmm.mrs.cpu import BytearrayMemoryResource, MallocMemoryResource
from devmm.testing import RecordingMemoryResource

STREAM = CpuStream()


def _deallocations(mr: RecordingMemoryResource) -> list[tuple[str, int, int, Stream]]:
    return [call for call in mr.calls if call[0] == "deallocate"]


class _OpaqueStream(Stream):
    """Minimal non-CPU stream: DeviceBuffer only reads `device` off it."""

    def __init__(self, device: Device) -> None:
        self.device = device

    @property
    def handle(self) -> int:
        return 0

    def synchronize(self) -> None:
        return None

    def wait_raw(self, other_handle: int) -> None:
        return None


class TestLifecycle:
    def test_construction_allocates_through_the_mr(
        self, recording_mr: RecordingMemoryResource
    ) -> None:
        buf = DeviceBuffer(64, mr=recording_mr, stream=STREAM)
        assert recording_mr.calls == [("allocate", buf.ptr, 64, STREAM)]
        assert buf.nbytes == 64
        assert buf.device == recording_mr.device
        assert buf.stream is STREAM
        assert buf.mr is recording_mr
        assert not buf.closed

    def test_free_deallocates_exactly_once_on_the_allocation_stream(
        self, recording_mr: RecordingMemoryResource
    ) -> None:
        buf = DeviceBuffer(64, mr=recording_mr, stream=STREAM)
        buf.free()
        assert buf.closed
        deallocations = _deallocations(recording_mr)
        assert deallocations == [("deallocate", buf.ptr, 64, STREAM)]
        assert deallocations[0][3] is STREAM

    def test_free_uses_the_explicit_stream_when_given(
        self, recording_mr: RecordingMemoryResource
    ) -> None:
        other = CpuStream()
        buf = DeviceBuffer(64, mr=recording_mr, stream=STREAM)
        buf.free(stream=other)
        deallocations = _deallocations(recording_mr)
        assert deallocations == [("deallocate", buf.ptr, 64, other)]
        assert deallocations[0][3] is other

    def test_second_free_is_a_noop(self, recording_mr: RecordingMemoryResource) -> None:
        buf = DeviceBuffer(64, mr=recording_mr, stream=STREAM)
        buf.free()
        buf.free()
        buf.free(stream=CpuStream())
        assert len(_deallocations(recording_mr)) == 1

    def test_free_with_explicit_stream_then_free_is_a_noop(
        self, recording_mr: RecordingMemoryResource
    ) -> None:
        buf = DeviceBuffer(64, mr=recording_mr, stream=STREAM)
        buf.free(stream=CpuStream())
        buf.free()
        assert len(_deallocations(recording_mr)) == 1

    def test_use_after_free_raises(self, recording_mr: RecordingMemoryResource) -> None:
        buf = DeviceBuffer(64, mr=recording_mr, stream=STREAM)
        buf.free()
        with pytest.raises(ValueError, match="freed"):
            buf.copy_to_host()
        with pytest.raises(ValueError, match="freed"):
            buf.copy_from_host(b"x")

    def test_context_manager_frees_on_exit(self, recording_mr: RecordingMemoryResource) -> None:
        with DeviceBuffer(64, mr=recording_mr, stream=STREAM) as buf:
            assert not buf.closed
        assert buf.closed
        assert len(_deallocations(recording_mr)) == 1

    def test_context_manager_frees_on_exception(
        self, recording_mr: RecordingMemoryResource
    ) -> None:
        with pytest.raises(RuntimeError, match="boom"):  # noqa: SIM117
            with DeviceBuffer(64, mr=recording_mr, stream=STREAM) as buf:
                raise RuntimeError("boom")
        assert buf.closed
        assert len(_deallocations(recording_mr)) == 1

    def test_entering_a_freed_buffer_raises(self, recording_mr: RecordingMemoryResource) -> None:
        buf = DeviceBuffer(64, mr=recording_mr, stream=STREAM)
        buf.free()
        with pytest.raises(ValueError, match="freed"), buf:
            pass

    def test_zero_byte_buffer_round_trips(self, recording_mr: RecordingMemoryResource) -> None:
        buf = DeviceBuffer(0, mr=recording_mr, stream=STREAM)
        buf.free()
        assert _deallocations(recording_mr) == [("deallocate", buf.ptr, 0, STREAM)]

    def test_mismatched_stream_and_mr_devices_raise(
        self, recording_mr: RecordingMemoryResource
    ) -> None:
        cuda = Device(DeviceType.CUDA)
        with pytest.raises(ValueError, match="device"):
            DeviceBuffer(64, mr=recording_mr, stream=_OpaqueStream(cuda))
        assert recording_mr.calls == []

    def test_free_on_a_foreign_device_stream_raises(
        self, recording_mr: RecordingMemoryResource
    ) -> None:
        buf = DeviceBuffer(64, mr=recording_mr, stream=STREAM)
        with pytest.raises(ValueError, match="device"):
            buf.free(stream=_OpaqueStream(Device(DeviceType.CUDA)))
        assert not buf.closed
        assert _deallocations(recording_mr) == []


class TestFinalizer:
    def test_dropping_all_refs_deallocates_exactly_once(
        self, recording_mr: RecordingMemoryResource
    ) -> None:
        buf = DeviceBuffer(64, mr=recording_mr, stream=STREAM)
        ptr = buf.ptr
        del buf
        gc.collect()
        assert _deallocations(recording_mr) == [("deallocate", ptr, 64, STREAM)]
        assert recording_mr.live == {}

    def test_buffer_in_a_reference_cycle_deallocates_exactly_once(
        self, recording_mr: RecordingMemoryResource
    ) -> None:
        buf = DeviceBuffer(64, mr=recording_mr, stream=STREAM)
        buf.cycle = buf  # type: ignore[attr-defined]
        ptr = buf.ptr
        del buf
        gc.collect()
        assert _deallocations(recording_mr) == [("deallocate", ptr, 64, STREAM)]

    def test_finalizer_does_not_resurrect_the_buffer(
        self, recording_mr: RecordingMemoryResource
    ) -> None:
        buf = DeviceBuffer(64, mr=recording_mr, stream=STREAM)
        buf.cycle = buf  # type: ignore[attr-defined]
        ref = weakref.ref(buf)
        del buf
        gc.collect()
        assert ref() is None
        assert len(_deallocations(recording_mr)) == 1

    def test_explicit_free_disarms_the_finalizer(
        self, recording_mr: RecordingMemoryResource
    ) -> None:
        buf = DeviceBuffer(64, mr=recording_mr, stream=STREAM)
        buf.free()
        del buf
        gc.collect()
        assert len(_deallocations(recording_mr)) == 1

    def test_interpreter_shutdown_without_free_exits_cleanly(self) -> None:
        program = textwrap.dedent(
            """
            from devmm import DeviceBuffer
            from devmm._core.stream import CpuStream
            from devmm.mrs.cpu import MallocMemoryResource

            buf = DeviceBuffer(1024, mr=MallocMemoryResource(), stream=CpuStream())
            buf.copy_from_host(b"\\xab" * 1024)
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", program], capture_output=True, text=True, timeout=60
        )
        assert result.returncode == 0
        assert result.stderr == ""


@pytest.fixture(params=[BytearrayMemoryResource, MallocMemoryResource], ids=["bytearray", "malloc"])
def cpu_mr(request: pytest.FixtureRequest) -> BytearrayMemoryResource | MallocMemoryResource:
    mr: BytearrayMemoryResource | MallocMemoryResource = request.param()
    return mr


class TestHostCopies:
    def test_round_trip_is_byte_exact(
        self, cpu_mr: BytearrayMemoryResource | MallocMemoryResource
    ) -> None:
        pattern = bytes(range(256)) * 4
        with DeviceBuffer(len(pattern), mr=cpu_mr, stream=STREAM) as buf:
            buf.copy_from_host(pattern)
            assert buf.copy_to_host() == pattern

    def test_round_trip_accepts_a_memoryview(
        self, cpu_mr: BytearrayMemoryResource | MallocMemoryResource
    ) -> None:
        pattern = bytearray(range(128))
        with DeviceBuffer(len(pattern), mr=cpu_mr, stream=STREAM) as buf:
            buf.copy_from_host(memoryview(pattern))
            assert buf.copy_to_host() == bytes(pattern)

    def test_partial_write_fills_a_prefix(
        self, cpu_mr: BytearrayMemoryResource | MallocMemoryResource
    ) -> None:
        with DeviceBuffer(8, mr=cpu_mr, stream=STREAM) as buf:
            buf.copy_from_host(b"\x00" * 8)
            buf.copy_from_host(b"\xff\xfe")
            assert buf.copy_to_host() == b"\xff\xfe" + b"\x00" * 6

    def test_oversized_write_raises(
        self, cpu_mr: BytearrayMemoryResource | MallocMemoryResource
    ) -> None:
        with DeviceBuffer(4, mr=cpu_mr, stream=STREAM) as buf:
            buf.copy_from_host(b"\xaa" * 4)
            with pytest.raises(ValueError, match="4-byte"):
                buf.copy_from_host(b"\x00" * 5)
            assert buf.copy_to_host() == b"\xaa" * 4

    def test_zero_byte_buffer_reads_back_empty(
        self, cpu_mr: BytearrayMemoryResource | MallocMemoryResource
    ) -> None:
        with DeviceBuffer(0, mr=cpu_mr, stream=STREAM) as buf:
            buf.copy_from_host(b"")
            assert buf.copy_to_host() == b""

    def test_non_contiguous_source_raises(
        self, cpu_mr: BytearrayMemoryResource | MallocMemoryResource
    ) -> None:
        strided = memoryview(bytes(range(16)))[::2]
        with (
            DeviceBuffer(16, mr=cpu_mr, stream=STREAM) as buf,
            pytest.raises(ValueError, match="contiguous"),
        ):
            buf.copy_from_host(strided)

    def test_fortran_contiguous_source_raises(
        self, cpu_mr: BytearrayMemoryResource | MallocMemoryResource
    ) -> None:
        # Fortran-contiguous views satisfy `memoryview.contiguous` but would
        # be silently reordered by a C-order byte copy; the guard must refuse
        # them, not just non-contiguous ones.
        np = pytest.importorskip("numpy")
        f_ordered = memoryview(np.asfortranarray(np.arange(12, dtype=np.uint8).reshape(3, 4)))
        assert f_ordered.contiguous and not f_ordered.c_contiguous
        with (
            DeviceBuffer(12, mr=cpu_mr, stream=STREAM) as buf,
            pytest.raises(ValueError, match="C-contiguous"),
        ):
            buf.copy_from_host(f_ordered)

    def test_copies_on_non_cpu_buffers_are_unsupported(self) -> None:
        cuda = Device(DeviceType.CUDA)
        mr = RecordingMemoryResource(device=cuda)
        buf = DeviceBuffer(16, mr=mr, stream=_OpaqueStream(cuda))
        with pytest.raises(NotImplementedError):
            buf.copy_to_host()
        with pytest.raises(NotImplementedError):
            buf.copy_from_host(b"\x00")
        buf.free()
