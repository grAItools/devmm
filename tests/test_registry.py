"""Registry contract tests: strong-ref current-MR storage keyed by device,
`using_memory_resource` restore-on-exit/-exception, contextvar isolation
across threads and asyncio tasks, and the unwired lazy default raising
cleanly (design §3.4).
"""

from __future__ import annotations

import asyncio
import gc
import re
import threading
import weakref
from collections.abc import Iterator

import pytest

from devmm import (
    Device,
    DeviceMemoryResource,
    DeviceType,
    get_current_memory_resource,
    set_current_memory_resource,
    using_memory_resource,
)
from devmm._core import registry as registry_module
from devmm.testing import RecordingMemoryResource

CPU = Device(DeviceType.CPU)
CUDA = Device(DeviceType.CUDA)


@pytest.fixture(autouse=True)
def _isolated_registry() -> Iterator[None]:
    """Snapshot and restore the process-wide registry around every test."""
    saved = dict(registry_module._registry)
    registry_module._registry.clear()
    yield
    registry_module._registry.clear()
    registry_module._registry.update(saved)


class TestSetAndGet:
    def test_set_then_get_returns_the_same_object(self) -> None:
        mr = RecordingMemoryResource()
        set_current_memory_resource(mr)
        assert get_current_memory_resource(CPU) is mr

    def test_set_is_keyed_by_the_mr_device(self) -> None:
        cpu_mr = RecordingMemoryResource(device=CPU)
        cuda_mr = RecordingMemoryResource(device=CUDA)
        set_current_memory_resource(cpu_mr)
        set_current_memory_resource(cuda_mr)
        assert get_current_memory_resource(CPU) is cpu_mr
        assert get_current_memory_resource(CUDA) is cuda_mr

    def test_set_replaces_the_previous_mr(self) -> None:
        first = RecordingMemoryResource()
        second = RecordingMemoryResource()
        set_current_memory_resource(first)
        set_current_memory_resource(second)
        assert get_current_memory_resource(CPU) is second

    def test_registry_holds_mrs_strongly(self) -> None:
        mr = RecordingMemoryResource()
        ref = weakref.ref(mr)
        set_current_memory_resource(mr)
        del mr
        gc.collect()
        assert ref() is not None
        assert get_current_memory_resource(CPU) is ref()

    def test_unset_device_raises_lookup_error_naming_the_device(self) -> None:
        with pytest.raises(LookupError, match=re.escape(str(CUDA))):
            get_current_memory_resource(CUDA)

    def test_failed_default_lookup_caches_nothing(self) -> None:
        with pytest.raises(LookupError):
            get_current_memory_resource(CPU)
        assert registry_module._registry == {}


class TestUsingMemoryResource:
    def test_override_wins_inside_and_restores_on_exit(self) -> None:
        base = RecordingMemoryResource()
        override = RecordingMemoryResource()
        set_current_memory_resource(base)
        with using_memory_resource(override):
            assert get_current_memory_resource(CPU) is override
        assert get_current_memory_resource(CPU) is base

    def test_override_restores_on_exception(self) -> None:
        base = RecordingMemoryResource()
        override = RecordingMemoryResource()
        set_current_memory_resource(base)
        with pytest.raises(RuntimeError, match="boom"), using_memory_resource(override):
            raise RuntimeError("boom")
        assert get_current_memory_resource(CPU) is base

    def test_using_yields_the_mr(self) -> None:
        override = RecordingMemoryResource()
        with using_memory_resource(override) as got:
            assert got is override

    def test_nested_overrides_unwind_in_order(self) -> None:
        base = RecordingMemoryResource()
        outer = RecordingMemoryResource()
        inner = RecordingMemoryResource()
        set_current_memory_resource(base)
        with using_memory_resource(outer):
            with using_memory_resource(inner):
                assert get_current_memory_resource(CPU) is inner
            assert get_current_memory_resource(CPU) is outer
        assert get_current_memory_resource(CPU) is base

    def test_override_is_scoped_to_the_mr_device(self) -> None:
        cpu_base = RecordingMemoryResource(device=CPU)
        cuda_override = RecordingMemoryResource(device=CUDA)
        set_current_memory_resource(cpu_base)
        with using_memory_resource(cuda_override):
            assert get_current_memory_resource(CPU) is cpu_base
            assert get_current_memory_resource(CUDA) is cuda_override

    def test_override_shadows_set_calls_made_inside_it(self) -> None:
        override = RecordingMemoryResource()
        late = RecordingMemoryResource()
        with using_memory_resource(override):
            set_current_memory_resource(late)
            assert get_current_memory_resource(CPU) is override
        assert get_current_memory_resource(CPU) is late


class TestIsolation:
    def test_overrides_are_isolated_across_threads(self) -> None:
        base = RecordingMemoryResource()
        main_override = RecordingMemoryResource()
        thread_override = RecordingMemoryResource()
        set_current_memory_resource(base)

        seen: list[DeviceMemoryResource] = []
        inside = threading.Event()
        release = threading.Event()

        def worker() -> None:
            seen.append(get_current_memory_resource(CPU))
            with using_memory_resource(thread_override):
                inside.set()
                release.wait(timeout=30)
                seen.append(get_current_memory_resource(CPU))

        with using_memory_resource(main_override):
            thread = threading.Thread(target=worker)
            thread.start()
            assert inside.wait(timeout=30)
            # The worker's override is live right now, yet invisible here.
            assert get_current_memory_resource(CPU) is main_override
            release.set()
            thread.join(timeout=30)

        assert seen == [base, thread_override]
        assert get_current_memory_resource(CPU) is base

    def test_overrides_are_isolated_across_asyncio_tasks(self) -> None:
        base = RecordingMemoryResource()
        first = RecordingMemoryResource()
        second = RecordingMemoryResource()
        set_current_memory_resource(base)

        async def scoped(mr: DeviceMemoryResource) -> DeviceMemoryResource:
            with using_memory_resource(mr):
                # Yield so both tasks hold their overrides concurrently.
                await asyncio.sleep(0)
                return get_current_memory_resource(CPU)

        async def main() -> tuple[DeviceMemoryResource, DeviceMemoryResource]:
            results = await asyncio.gather(scoped(first), scoped(second))
            # Task overrides never leak into the awaiting context.
            assert get_current_memory_resource(CPU) is base
            return results[0], results[1]

        got_first, got_second = asyncio.run(main())
        assert got_first is first
        assert got_second is second
        assert get_current_memory_resource(CPU) is base
