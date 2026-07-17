"""Self-tests for the public conformance entry points (design §9):
`devmm.testing.mr_conformance` and `devmm.testing.dlpack_conformance` pass
against the reference CPU implementations and fail loudly on doubles that
violate the contracts they pin.
"""

from __future__ import annotations

import ctypes

import pytest

from devmm import (
    Device,
    DeviceMemoryResource,
    StatisticsAdaptor,
    Stream,
    Tensor,
    using_memory_resource,
)
from devmm._core.stream import CpuStream
from devmm.mrs.cpu import BytearrayMemoryResource, MallocMemoryResource
from devmm.testing import dlpack_conformance, mr_conformance

_CPU = Device.from_string("cpu")


def _bytearray_factory(*, alignment: int | None = None) -> BytearrayMemoryResource:
    if alignment is None:
        return BytearrayMemoryResource()
    return BytearrayMemoryResource(alignment=alignment)


def _malloc_factory(*, alignment: int | None = None) -> MallocMemoryResource:
    if alignment is None:
        return MallocMemoryResource()
    return MallocMemoryResource(alignment=alignment)


class _BumpAllocatorMR(DeviceMemoryResource):
    """A minimal, correct host bump allocator the broken doubles derive from.

    Never reuses addresses, so double-free/foreign-free detection can be a
    plain live-set lookup.
    """

    def __init__(self) -> None:
        self.device = _CPU
        self._backing = (ctypes.c_char * (1 << 20))()
        self._cursor = ctypes.addressof(self._backing)
        self._live: set[int] = set()

    def allocate(self, nbytes: int, stream: Stream) -> int:
        if nbytes < 0:
            raise ValueError(f"cannot allocate a negative size ({nbytes} bytes)")
        remainder = self._cursor % 64
        if remainder:
            self._cursor += 64 - remainder
        ptr = self._cursor
        self._cursor += max(nbytes, 1)
        self._live.add(ptr)
        return ptr

    def deallocate(self, ptr: int, nbytes: int, stream: Stream) -> None:
        if ptr not in self._live:
            raise ValueError(f"pointer {ptr:#x} is not a live allocation")
        self._live.discard(ptr)

    def guaranteed_alignment(self) -> int:
        return 64

    def _debug_live_count(self) -> int:
        return len(self._live)


class _AliasingMR(_BumpAllocatorMR):
    """Contract violation: every allocation shares one address."""

    def allocate(self, nbytes: int, stream: Stream) -> int:
        if nbytes < 0:
            raise ValueError(f"cannot allocate a negative size ({nbytes} bytes)")
        self._live.add(ctypes.addressof(self._backing))
        return ctypes.addressof(self._backing)

    def deallocate(self, ptr: int, nbytes: int, stream: Stream) -> None:
        return None


class _MisalignedMR(_BumpAllocatorMR):
    """Contract violation: claims 64-byte alignment, hands out odd pointers."""

    def allocate(self, nbytes: int, stream: Stream) -> int:
        return super().allocate(nbytes, stream) + 1

    def deallocate(self, ptr: int, nbytes: int, stream: Stream) -> None:
        super().deallocate(ptr - 1, nbytes, stream)


class _UndetectedDoubleFreeMR(_BumpAllocatorMR):
    """Contract violation: deallocate never rejects misuse."""

    def deallocate(self, ptr: int, nbytes: int, stream: Stream) -> None:
        self._live.discard(ptr)


class _HookLessMR(_BumpAllocatorMR):
    """Contract violation: no `_debug_live_count()` testing hook."""

    _debug_live_count = None  # type: ignore[assignment]


class _StringCountMR(_BumpAllocatorMR):
    """Contract violation: the testing hook returns a non-int."""

    def _debug_live_count(self) -> int:
        return str(len(self._live))  # type: ignore[return-value]


class _UninspectableFactory:
    """A factory `inspect.signature` cannot introspect: the alignment sweep
    must be skipped, not crash."""

    @property
    def __signature__(self) -> object:
        raise ValueError("no signature")

    def __call__(self) -> MallocMemoryResource:
        return MallocMemoryResource()


class TestMrConformance:
    @pytest.mark.parametrize("factory", (_bytearray_factory, _malloc_factory))
    def test_passes_for_the_reference_cpu_mrs(
        self, factory: type[BytearrayMemoryResource] | type[MallocMemoryResource]
    ) -> None:
        mr_conformance(factory)

    def test_alignment_sweep_needs_no_alignment_keyword(self) -> None:
        # A factory without the optional `alignment` keyword still runs the
        # rest of the suite; the requested-alignment sweep is skipped.
        mr_conformance(lambda: MallocMemoryResource())

    def test_detects_aliasing_allocations(self) -> None:
        with pytest.raises(AssertionError, match="alias"):
            mr_conformance(_AliasingMR)

    def test_detects_a_dishonest_guaranteed_alignment(self) -> None:
        with pytest.raises(AssertionError, match="alignment"):
            mr_conformance(_MisalignedMR)

    def test_detects_an_undetected_double_free(self) -> None:
        with pytest.raises(AssertionError, match="double-free"):
            mr_conformance(_UndetectedDoubleFreeMR)

    def test_requires_the_debug_live_count_hook(self) -> None:
        with pytest.raises(AssertionError, match="_debug_live_count"):
            mr_conformance(_HookLessMR)

    def test_requires_an_integer_live_count(self) -> None:
        with pytest.raises(AssertionError, match="must return an int"):
            mr_conformance(_StringCountMR)

    def test_an_uninspectable_factory_skips_the_alignment_sweep(self) -> None:
        mr_conformance(_UninspectableFactory())

    def test_injected_write_read_and_stream_factory_are_used(self) -> None:
        writes: list[int] = []
        reads: list[int] = []
        streams: list[Stream] = []

        def write(ptr: int, data: bytes) -> None:
            writes.append(ptr)
            ctypes.memmove(ptr, data, len(data))

        def read(ptr: int, nbytes: int) -> bytes:
            reads.append(ptr)
            return ctypes.string_at(ptr, nbytes)

        def stream_factory() -> Stream:
            stream = CpuStream()
            streams.append(stream)
            return stream

        mr_conformance(_malloc_factory, stream_factory=stream_factory, write=write, read=read)
        assert writes and reads and streams


class TestDlpackConformance:
    def test_passes_on_cpu(self) -> None:
        dlpack_conformance(_CPU)

    def test_exercises_the_current_registry_mr(self) -> None:
        stats = StatisticsAdaptor(MallocMemoryResource())
        with using_memory_resource(stats):
            dlpack_conformance(_CPU)
        assert stats.total_bytes > 0
        assert stats.current_bytes == 0

    def test_rejects_a_non_device_argument(self) -> None:
        with pytest.raises(TypeError):
            dlpack_conformance("cpu")  # type: ignore[arg-type]

    def test_detects_a_broken_dlpack_device(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(Tensor, "__dlpack_device__", lambda self: (1, 99))
        with pytest.raises(AssertionError, match="__dlpack_device__"):
            dlpack_conformance(_CPU)
