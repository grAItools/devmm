"""Deleter lifecycle matrix (design §7.2/§7.4), all against the recording MR:
(a) consumed capsules defer the free to the consumer, (b) unconsumed capsules
release their holder, (c) double-exports free exactly once, (d) a refleak
harness stays bounded, (e) deleters run safely on foreign threads, and
(f) consumers outliving module teardown shut down cleanly.
"""

from __future__ import annotations

import ctypes
import gc
import subprocess
import sys
import textwrap
import threading
import tracemalloc

import numpy as np
import pytest

from devmm import Stream, Tensor, empty
from devmm._dlpack._abi import DLManagedTensorVersioned
from devmm.testing import RecordingMemoryResource

_capsule_get_pointer = ctypes.PYFUNCTYPE(ctypes.c_void_p, ctypes.py_object, ctypes.c_char_p)(
    ("PyCapsule_GetPointer", ctypes.pythonapi)
)


def _deallocations(mr: RecordingMemoryResource) -> list[tuple[str, int, int, Stream]]:
    return [call for call in mr.calls if call[0] == "deallocate"]


def _tensor(mr: RecordingMemoryResource) -> Tensor:
    return empty((4,), "float32", mr=mr)


def test_consumed_capsule_defers_the_free_to_the_consumer(
    recording_mr: RecordingMemoryResource,
) -> None:
    t = _tensor(recording_mr)
    consumer = np.from_dlpack(t)
    del t
    gc.collect()
    assert _deallocations(recording_mr) == []
    del consumer
    gc.collect()
    assert len(_deallocations(recording_mr)) == 1


@pytest.mark.parametrize("max_version", [None, (1, 1)], ids=["legacy", "versioned"])
def test_unconsumed_capsule_releases_the_holder(
    recording_mr: RecordingMemoryResource, max_version: tuple[int, int] | None
) -> None:
    t = _tensor(recording_mr)
    capsule = t.__dlpack__(max_version=max_version)
    del capsule
    gc.collect()
    # The capsule destructor released the holder, but the user still holds
    # the tensor, so the allocation stays live.
    assert _deallocations(recording_mr) == []
    del t
    gc.collect()
    assert len(_deallocations(recording_mr)) == 1


def test_double_export_frees_exactly_once_after_the_last_consumer(
    recording_mr: RecordingMemoryResource,
) -> None:
    t = _tensor(recording_mr)
    first = np.from_dlpack(t)
    second = np.from_dlpack(t)
    del t
    gc.collect()
    assert _deallocations(recording_mr) == []
    del first
    gc.collect()
    assert _deallocations(recording_mr) == []
    del second
    gc.collect()
    assert len(_deallocations(recording_mr)) == 1


def test_deleter_thunk_survives_gc_between_export_and_consumption(
    recording_mr: RecordingMemoryResource,
) -> None:
    t = _tensor(recording_mr)
    capsule = t.__dlpack__(max_version=(1, 1))
    gc.collect()
    # Destroying the capsule invokes the module-scope deleter thunk; a
    # collected thunk would crash here instead of releasing the holder.
    del capsule
    del t
    gc.collect()
    assert len(_deallocations(recording_mr)) == 1


def test_a_second_deleter_call_is_a_noop(recording_mr: RecordingMemoryResource) -> None:
    t = _tensor(recording_mr)
    capsule = t.__dlpack__(max_version=(1, 1))
    managed = ctypes.cast(
        _capsule_get_pointer(capsule, b"dltensor_versioned"),
        ctypes.POINTER(DLManagedTensorVersioned),
    )
    ctx = managed.contents.manager_ctx
    assert ctx is not None
    # Hold the holder directly so its block outlives the first deleter call
    # and the second call reads the nulled manager_ctx, not freed memory.
    holder = ctypes.cast(ctx, ctypes.py_object).value
    managed.contents.deleter(managed)
    assert managed.contents.manager_ctx is None
    # Protocol violation: consumers must call the deleter exactly once; a
    # second call must not double-decref the holder.
    managed.contents.deleter(managed)
    del capsule  # the capsule destructor's own deleter call is a no-op too
    gc.collect()
    assert _deallocations(recording_mr) == []
    del managed, holder, t
    gc.collect()
    assert len(_deallocations(recording_mr)) == 1


def test_refleak_harness_stays_bounded_over_many_iterations() -> None:
    def once() -> None:
        mr = RecordingMemoryResource()
        t = empty((8,), "float32", mr=mr)
        consumer = np.from_dlpack(t)
        del t, consumer

    for _ in range(100):  # warm caches before measuring
        once()
    gc.collect()
    gc.garbage.clear()
    tracemalloc.start()
    try:
        for _ in range(10_000):
            once()
        gc.collect()
        current, _peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert current < 256 * 1024
    assert gc.garbage == []


def test_deleter_runs_on_a_foreign_thread(recording_mr: RecordingMemoryResource) -> None:
    t = _tensor(recording_mr)
    box = [np.from_dlpack(t)]
    del t
    gc.collect()
    assert _deallocations(recording_mr) == []

    # Dropping the last consumer reference off the main thread runs the
    # ctypes deleter there; the callback must acquire the GIL, not deadlock.
    thread = threading.Thread(target=box.clear)
    thread.start()
    thread.join(timeout=30)
    assert not thread.is_alive()
    gc.collect()
    assert len(_deallocations(recording_mr)) == 1


def test_consumer_outliving_module_teardown_exits_cleanly() -> None:
    program = textwrap.dedent(
        """
        import numpy as np

        import devmm
        from devmm.mrs.cpu import MallocMemoryResource

        tensor = devmm.empty((16,), "float32", mr=MallocMemoryResource())
        consumer = np.from_dlpack(tensor)
        del tensor
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0
    assert result.stderr == ""
