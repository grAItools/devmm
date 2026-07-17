"""Scripted third-party doubles for the integration arrows (design §6, §9):
a `cupy` module fake exposing the allocator hook registry and recording the
device/stream contexts active around allocator calls, a `numba.cuda` module
fake with the EMM surface `DevmmEMMPlugin` subclasses and the
`set_memory_manager` global it mutates, and an `rmm`-named module fake with
the per-device resource registry the rmm bridge drives.
"""

from __future__ import annotations

from collections.abc import Callable
from types import ModuleType, TracebackType
from typing import Any


class FakeCupyStream:
    """A CuPy stream double: the raw handle under CuPy's `.ptr` spelling."""

    def __init__(self, ptr: int) -> None:
        self.ptr = ptr


class FakeCupyMemory:
    """The raw block behind a fake `MemoryPointer`."""

    def __init__(self, ptr: int, size: int) -> None:
        self.ptr = ptr
        self.size = size


class FakeCupyMemoryPointer:
    """CuPy `MemoryPointer` double: `.ptr` is `mem.ptr + offset`."""

    def __init__(self, mem: Any, offset: int) -> None:
        self.mem = mem
        self.ptr = int(mem.ptr) + offset


class FakeCupyUnownedMemory:
    """CuPy `UnownedMemory` double: remembers what it wrapped."""

    def __init__(self, ptr: int, size: int, owner: Any, device_id: int) -> None:
        self.ptr = ptr
        self.size = size
        self.owner = owner
        self.device_id = device_id


class _FakeCupyScope:
    """Context manager double for `cupy.cuda.Device`/`ExternalStream`:
    records enter/exit in the module's event log and flips the module's
    fake thread-local current device/stream for the scope."""

    def __init__(self, module: FakeCupy, kind: str, value: int) -> None:
        self._module = module
        self._kind = kind
        self._value = value
        self._previous: int | None = None

    def __enter__(self) -> _FakeCupyScope:
        module = self._module
        module.events.append((f"{self._kind}_enter", self._value))
        if self._kind == "device":
            self._previous = module.current_device
            module.current_device = self._value
        else:
            self._previous = module.current_stream_ptr
            module.current_stream_ptr = self._value
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        module = self._module
        module.events.append((f"{self._kind}_exit", self._value))
        if self._kind == "device":
            module.current_device = self._previous
        else:
            module.current_stream_ptr = self._previous


class FakeCupyCudaNamespace:
    """The `cupy.cuda` attribute surface devmm touches."""

    def __init__(self, module: FakeCupy) -> None:
        self._module = module
        self.UnownedMemory = FakeCupyUnownedMemory
        self.MemoryPointer = FakeCupyMemoryPointer
        self.current_stream = FakeCupyStream(0)
        # A stable default-allocator object so identity assertions across
        # get/set round trips behave like CuPy's own module-level default.
        self._default_allocator: Callable[[int], Any] = self.alloc
        self._allocator: Callable[[int], Any] = self._default_allocator

    def Device(self, index: int) -> _FakeCupyScope:
        return _FakeCupyScope(self._module, "device", index)

    def ExternalStream(self, ptr: int) -> _FakeCupyScope:
        return _FakeCupyScope(self._module, "stream", ptr)

    def get_current_stream(self) -> FakeCupyStream:
        return self.current_stream

    def alloc(self, nbytes: int) -> FakeCupyMemoryPointer:
        module = self._module
        module.events.append(("alloc", nbytes, module.current_device, module.current_stream_ptr))
        return FakeCupyMemoryPointer(FakeCupyMemory(module.take_ptr(nbytes), nbytes), 0)

    def set_allocator(self, allocator: Callable[[int], Any] | None = None) -> None:
        self._module.events.append(("set_allocator", allocator))
        self._allocator = self._default_allocator if allocator is None else allocator

    def get_allocator(self) -> Callable[[int], Any]:
        return self._allocator


class FakeCupy:
    """The `cupy` module double.

    `events` records, in order, context enters/exits, allocator calls (with
    the fake current device/stream active at call time) and
    `set_allocator` calls, so tests can assert both the hook registry and
    the context discipline around allocations.
    """

    def __init__(self) -> None:
        self.events: list[tuple[Any, ...]] = []
        self.current_device: int | None = None
        self.current_stream_ptr: int | None = None
        self._next_ptr = 0x40000
        self.cuda = FakeCupyCudaNamespace(self)

    def take_ptr(self, nbytes: int) -> int:
        """Deterministic 256-aligned, never-reused fake device pointers."""
        ptr = self._next_ptr
        self._next_ptr += -(-max(nbytes, 1) // 256) * 256
        return ptr


class FakeNumbaMemoryPointer:
    """`numba.cuda.MemoryPointer` double: frees exactly once through the
    finalizer, as Numba's deallocation machinery does."""

    def __init__(
        self,
        context: Any,
        pointer: Any,
        size: int,
        owner: Any = None,
        finalizer: Callable[[], None] | None = None,
    ) -> None:
        self.context = context
        self.pointer = pointer
        self.size = size
        self.owner = owner
        self._finalizer = finalizer

    def free(self) -> None:
        finalizer, self._finalizer = self._finalizer, None
        if finalizer is not None:
            finalizer()


class FakeNumbaMemoryInfo:
    """`numba.cuda.MemoryInfo` double."""

    def __init__(self, free: int, total: int) -> None:
        self.free = free
        self.total = total


class FakeHostOnlyCUDAMemoryManager:
    """`numba.cuda.HostOnlyCUDAMemoryManager` double: the base-class surface
    the EMM protocol guarantees — context binding plus the `allocations`
    mapping plugins stash owners in."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if "context" not in kwargs:
            raise RuntimeError("BaseCUDAMemoryManager requires a context keyword argument")
        self.context = kwargs.pop("context")
        self.allocations: dict[int, Any] = {}


class FakeGetIpcHandleMixin:
    """`numba.cuda.GetIpcHandleMixin` double."""

    def get_ipc_handle(self, memory: Any) -> Any:
        return ("ipc", memory)


class FakeNumbaDriver:
    """`numba.cuda.cudadrv.driver` double: the memory-manager global that
    `set_memory_manager` writes and uninstall restores."""

    def __init__(self) -> None:
        self._memory_manager: type | None = None


class _FakeNumbaCudadrv:
    def __init__(self) -> None:
        self.driver = FakeNumbaDriver()


class FakeNumbaCuda(ModuleType):
    """The `numba.cuda` module double: EMM base classes and the
    `set_memory_manager` validation Numba performs (instantiate with
    `context=None`, check `interface_version`, then commit the global)."""

    def __init__(self) -> None:
        super().__init__("numba.cuda")
        self.HostOnlyCUDAMemoryManager = FakeHostOnlyCUDAMemoryManager
        self.GetIpcHandleMixin = FakeGetIpcHandleMixin
        self.MemoryPointer = FakeNumbaMemoryPointer
        self.MemoryInfo = FakeNumbaMemoryInfo
        self.cudadrv = _FakeNumbaCudadrv()
        self.memory_manager_calls: list[type] = []

    def set_memory_manager(self, mm_plugin: type) -> None:
        self.memory_manager_calls.append(mm_plugin)
        dummy = mm_plugin(context=None)
        if dummy.interface_version != 1:
            raise RuntimeError(
                f"EMM Plugin interface has version {dummy.interface_version} - version 1 required"
            )
        self.cudadrv.driver._memory_manager = mm_plugin


class FakeRmmInitialResource:
    """Stands in for the resource rmm's registry serves before install."""

    def __init__(self, index: int) -> None:
        self.index = index


class FakeRmmCallbackMemoryResource:
    """`rmm.mr.CallbackMemoryResource` double: stores the callbacks so tests
    can drive them exactly as rmm's trampoline would."""

    def __init__(
        self,
        allocate_func: Callable[..., int],
        deallocate_func: Callable[..., None],
    ) -> None:
        self.allocate_func = allocate_func
        self.deallocate_func = deallocate_func


class FakeRmmMr:
    """The `rmm.mr` namespace: one platform marker class (design §4.2) plus
    the per-device resource registry the bridge mutates."""

    def __init__(self, marker: str) -> None:
        setattr(self, marker, type(marker, (), {}))
        self.CallbackMemoryResource = FakeRmmCallbackMemoryResource
        self.per_device: dict[int, Any] = {}
        self.calls: list[tuple[Any, ...]] = []

    def get_per_device_resource(self, index: int) -> Any:
        self.calls.append(("get_per_device_resource", index))
        return self.per_device.setdefault(index, FakeRmmInitialResource(index))

    def set_per_device_resource(self, index: int, resource: Any) -> None:
        self.calls.append(("set_per_device_resource", index, resource))
        self.per_device[index] = resource


def fake_rmm_module(marker: str) -> ModuleType:
    """An `rmm`-named module double carrying `marker` on its `mr` namespace,
    ready for `monkeypatch.setitem(sys.modules, "rmm", ...)`."""
    module = ModuleType("rmm")
    module.mr = FakeRmmMr(marker)  # type: ignore[attr-defined]
    return module


class FakeForeignStream:
    """A foreign stream object speaking the CUDA stream protocol."""

    def __init__(self, handle: int) -> None:
        self._handle = handle

    def __cuda_stream__(self) -> tuple[int, int]:
        return (0, self._handle)
