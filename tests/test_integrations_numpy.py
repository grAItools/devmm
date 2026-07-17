"""`integrations.numpy.install` (NEP-49, design §6) against the real NumPy:
the ctypes `PyDataMem_Handler` mirror vs the NumPy-version-parametrized
offset table (plus decoding the live default handler as an oracle), the
install -> allocate -> statistics -> uninstall round trip through NumPy's
allocation entry points (malloc/calloc/realloc), per-array handler retention
after uninstall, the direct-cycle guard and the supported-range guard.
"""

from __future__ import annotations

import contextlib
import ctypes
import gc
import importlib
import logging
import sys
from collections.abc import Iterator
from typing import Any

import pytest

from devmm import (
    CallbackMemoryResource,
    Device,
    DeviceMemoryResource,
    LimitingAdaptor,
    StatisticsAdaptor,
    Stream,
    _nep49,
)
from devmm.integrations import numpy as integrations_numpy
from devmm.mrs.cpu import MallocMemoryResource, NumpyHandlerMemoryResource
from devmm.testing import RecordingMemoryResource

np = pytest.importorskip("numpy")

_CPU = Device.from_string("cpu")

_POINTER = ctypes.sizeof(ctypes.c_void_p)


def _get_handler_name(*args: object) -> str:
    # NumPy 2.x moved multiarray to `numpy._core`; 1.x keeps `numpy.core`.
    try:
        multiarray = importlib.import_module("numpy._core.multiarray")
    except ImportError:
        multiarray = importlib.import_module("numpy.core.multiarray")
    name = multiarray.get_handler_name(*args)
    assert name is not None
    return str(name)


def _stats() -> StatisticsAdaptor:
    return StatisticsAdaptor(MallocMemoryResource())


# NumPy-version-parametrized expectations for the `PyDataMem_Handler` mirror
# on 64-bit ABIs: [low, high) version ranges -> struct size and field
# offsets. The struct has been stable since NEP-49 landed in 1.22; a future
# NumPy that changes it needs a new row here *and* a new mirror before the
# range guard admits it.
_OFFSET_TABLE: list[tuple[tuple[int, int], tuple[int, int], dict[str, int]]] = [
    (
        (1, 22),
        (3, 0),
        {
            "sizeof": 168,
            "name": 0,
            "version": 127,
            "allocator": 128,
            "allocator.sizeof": 40,
            "allocator.ctx": 0,
            "allocator.malloc": 8,
            "allocator.calloc": 16,
            "allocator.realloc": 24,
            "allocator.free": 32,
        },
    ),
]


@pytest.mark.skipif(_POINTER != 8, reason="the offset table covers 64-bit ABIs")
class TestHandlerMirror:
    def test_mirror_matches_the_version_parametrized_offset_table(self) -> None:
        version = _nep49.parsed_version(np.__version__)
        rows = [expected for low, high, expected in _OFFSET_TABLE if low <= version < high]
        assert rows, f"no NEP-49 offset row covers numpy {np.__version__}; extend the table"
        expected = rows[0]
        assert ctypes.sizeof(_nep49.PyDataMemHandler) == expected["sizeof"]
        assert ctypes.sizeof(_nep49.PyDataMemAllocator) == expected["allocator.sizeof"]
        for field in ("name", "version", "allocator"):
            assert getattr(_nep49.PyDataMemHandler, field).offset == expected[field]
        for field in ("ctx", "malloc", "calloc", "realloc", "free"):
            offset = getattr(_nep49.PyDataMemAllocator, field).offset
            assert offset == expected[f"allocator.{field}"]

    def test_entry_points_are_refcount_neutral(self) -> None:
        # ctypes' py_object restype takes ownership of the new reference
        # both entry points return; an extra incref/decref in the wrappers
        # would leak — or worse, free — NumPy's live handler capsule.
        api = _nep49.load_api()
        capsule = api.get_handler()
        base = sys.getrefcount(capsule)
        for _ in range(1000):
            api.get_handler()
        assert sys.getrefcount(capsule) == base

        def cycle(count: int) -> None:
            for _ in range(count):
                previous = api.set_handler(capsule)
                api.set_handler(previous)

        # NumPy-side context machinery retains a couple of one-off cached
        # references the first times a capsule is installed; warm up until
        # steady state, then pin that the steady state is exactly neutral.
        cycle(10)
        base = sys.getrefcount(capsule)
        cycle(1000)
        assert sys.getrefcount(capsule) == base

    def test_mirror_decodes_the_live_default_handler(self) -> None:
        # Oracle beyond self-agreement: read NumPy's own static handler
        # through the mirror — wrong offsets would decode garbage.
        api = _nep49.load_api()
        capsule = api.get_handler()
        handler = _nep49.handler_pointer(capsule).contents
        assert handler.name == b"default_allocator"
        assert handler.version == _nep49.HANDLER_ABI_VERSION


class TestInstallRoundTrip:
    def test_install_grows_statistics_and_uninstall_restores_the_prior_handler(self) -> None:
        prior = _get_handler_name()
        stats = _stats()
        handle = integrations_numpy.install(stats)
        try:
            assert _get_handler_name() == "devmm"
            array = np.empty(8192, dtype=np.uint8)
            assert stats.current_bytes == 8192
            array[:] = 7
            assert int(array.sum()) == 7 * 8192
            del array
            assert stats.current_bytes == 0
        finally:
            handle.uninstall()
        assert _get_handler_name() == prior

    def test_context_manager_restores_on_exception(self) -> None:
        prior = _get_handler_name()
        with pytest.raises(RuntimeError, match="boom"):  # noqa: SIM117 - the point is nesting
            with integrations_numpy.install(_stats()):
                assert _get_handler_name() == "devmm"
                raise RuntimeError("boom")
        assert _get_handler_name() == prior

    def test_calloc_path_allocates_zeroed_memory_through_the_mr(self) -> None:
        stats = _stats()
        with integrations_numpy.install(stats):
            array = np.zeros(4096, dtype=np.uint8)
            assert stats.current_bytes == 4096
            assert int(array.sum()) == 0
            del array
            assert stats.current_bytes == 0

    def test_realloc_path_preserves_the_prefix_and_retracks_the_size(self) -> None:
        stats = _stats()
        with integrations_numpy.install(stats):
            array = np.arange(64, dtype=np.int64)
            assert stats.current_bytes == 512
            array.resize(128, refcheck=False)
            assert stats.current_bytes == 1024
            assert list(array[:64]) == list(range(64))
            del array
            assert stats.current_bytes == 0

    def test_allocation_failure_surfaces_as_numpy_memory_error(self) -> None:
        prior = _get_handler_name()
        limited = LimitingAdaptor(MallocMemoryResource(), limit_bytes=1024)
        with integrations_numpy.install(limited), pytest.raises(MemoryError):
            np.empty(1_000_000, dtype=np.uint8)
        assert _get_handler_name() == prior

    def test_uninstall_restores_a_previous_devmm_handler(self) -> None:
        first_stats = _stats()
        second_stats = _stats()
        first = integrations_numpy.install(first_stats)
        try:
            second = integrations_numpy.install(second_stats)
            array = np.empty(2048, dtype=np.uint8)
            assert second_stats.current_bytes == 2048
            assert first_stats.current_bytes == 0
            del array
            second.uninstall()
            array = np.empty(2048, dtype=np.uint8)
            assert first_stats.current_bytes == 2048
            assert second_stats.current_bytes == 0
            del array
        finally:
            first.uninstall()


class TestPerArrayHandlerRetention:
    def test_arrays_allocated_during_install_stay_freeable_after(self) -> None:
        prior = _get_handler_name()
        live_states_before = len(integrations_numpy._LIVE)
        stats = _stats()
        handle = integrations_numpy.install(stats)
        assert len(integrations_numpy._LIVE) == live_states_before + 1
        array = np.empty(4096, dtype=np.uint8)
        handle.uninstall()
        del handle
        gc.collect()
        assert _get_handler_name() == prior
        # NumPy keeps the allocating handler per array, so the free still
        # routes through the devmm MR — long after uninstall (design §6),
        # and the array's capsule reference keeps the handler state alive.
        assert _get_handler_name(array) == "devmm"
        assert len(integrations_numpy._LIVE) == live_states_before + 1
        assert stats.current_bytes == 4096
        array[:] = 3
        assert int(array.sum()) == 3 * 4096
        del array
        gc.collect()
        assert stats.current_bytes == 0
        # The last capsule reference died with the array: the capsule
        # destructor must have retired the handler state, or every install
        # would leak a _HandlerState + MR invisibly.
        assert len(integrations_numpy._LIVE) == live_states_before


class TestRefusals:
    def test_composing_the_consume_and_provide_arrows_raises(self) -> None:
        with pytest.raises(ValueError, match="cycle"):
            integrations_numpy.install(NumpyHandlerMemoryResource())

    def test_the_cycle_is_detected_through_the_adaptor_chain(self) -> None:
        with pytest.raises(ValueError, match="cycle"):
            integrations_numpy.install(StatisticsAdaptor(NumpyHandlerMemoryResource()))

    def test_non_cpu_mrs_are_rejected(self) -> None:
        mr = RecordingMemoryResource(Device.from_string("cuda:0"))
        with pytest.raises(ValueError, match="cpu"):
            integrations_numpy.install(mr)

    @pytest.mark.parametrize("version", ["1.21.6", "3.0.0"])
    def test_out_of_range_numpy_raises_and_leaves_the_handler_alone(
        self, monkeypatch: pytest.MonkeyPatch, version: str
    ) -> None:
        prior = _get_handler_name()
        monkeypatch.setattr(np, "__version__", version)
        with pytest.raises(RuntimeError, match=version.replace(".", r"\.")):
            integrations_numpy.install(_stats())
        assert _get_handler_name() == prior


class TestThunkFailSafety:
    """The module-global allocator thunks are C entry points: a Python
    exception cannot cross into NumPy, so unknown contexts, unknown pointers
    and failing MRs must degrade to NULL returns / logged no-ops."""

    @contextlib.contextmanager
    def _registered(self, mr: DeviceMemoryResource) -> Iterator[Any]:
        state = integrations_numpy._HandlerState(mr)
        integrations_numpy._LIVE[state.address] = state
        try:
            yield state
        finally:
            integrations_numpy._LIVE.pop(state.address, None)

    def test_thunks_with_an_unknown_ctx_fail_safe(self) -> None:
        assert integrations_numpy._malloc_impl(0xDEAD, 16) == 0
        assert integrations_numpy._calloc_impl(0xDEAD, 4, 4) == 0
        assert integrations_numpy._realloc_impl(0xDEAD, 0x1234, 16) == 0
        integrations_numpy._free_impl(0xDEAD, 0x1234, 16)

    def test_free_of_a_null_or_unknown_pointer_is_a_logged_noop(self) -> None:
        with self._registered(MallocMemoryResource()) as state:
            integrations_numpy._free_impl(state.address, 0, 8)
            integrations_numpy._free_impl(state.address, 0x999, 8)
            assert state.sizes == {}

    def test_realloc_of_a_null_pointer_is_a_plain_allocation(self) -> None:
        with self._registered(MallocMemoryResource()) as state:
            ptr = integrations_numpy._realloc_impl(state.address, 0, 32)
            assert ptr != 0
            assert state.sizes == {ptr: 32}
            integrations_numpy._free_impl(state.address, ptr, 32)
            assert state.sizes == {}

    def test_realloc_of_an_unknown_pointer_returns_null(self) -> None:
        with self._registered(MallocMemoryResource()) as state:
            assert integrations_numpy._realloc_impl(state.address, 0x999, 32) == 0

    def test_failed_realloc_keeps_the_old_allocation_valid(self) -> None:
        with self._registered(LimitingAdaptor(MallocMemoryResource(), limit_bytes=64)) as state:
            ptr = integrations_numpy._malloc_impl(state.address, 48)
            assert ptr != 0
            # Growing to 128 exceeds the 64-byte budget: C realloc semantics
            # say fail with NULL and leave the old allocation live.
            assert integrations_numpy._realloc_impl(state.address, ptr, 128) == 0
            assert state.sizes == {ptr: 48}
            integrations_numpy._free_impl(state.address, ptr, 48)

    def test_an_unexpected_allocate_exception_is_logged_and_returns_null(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        def broken_alloc(nbytes: int, stream: Stream) -> int:
            raise RuntimeError("boom")

        mr = CallbackMemoryResource(broken_alloc, lambda p, n, s: None, _CPU)
        with (
            self._registered(mr) as state,
            caplog.at_level(logging.ERROR, logger="devmm.mr"),
        ):
            assert integrations_numpy._malloc_impl(state.address, 16) == 0
        assert any("allocating" in record.message for record in caplog.records)

    def test_a_deallocate_exception_is_logged_and_swallowed(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        def broken_dealloc(ptr: int, nbytes: int, stream: Stream) -> None:
            raise RuntimeError("boom")

        # The thunks never dereference what the MR returns, so a fake
        # pointer is safe here.
        mr = CallbackMemoryResource(lambda n, s: 0x1000, broken_dealloc, _CPU)
        with (
            self._registered(mr) as state,
            caplog.at_level(logging.ERROR, logger="devmm.mr"),
        ):
            ptr = integrations_numpy._malloc_impl(state.address, 16)
            integrations_numpy._free_impl(state.address, ptr, 16)
            assert state.sizes == {}
        assert any("freeing" in record.message for record in caplog.records)

    def test_destroying_a_null_capsule_pointer_is_a_noop(self) -> None:
        integrations_numpy._destroy_handler_capsule(None)
