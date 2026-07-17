"""Shared GPU FFI shim (design §4.2): one implementation of the runtime SPI,
the raw-runtime MR and the rmm-module wrapper for both GPU platforms.

The CUDA and HIP runtime C APIs are signature-identical for every entry
point devmm touches (hipMM/HIP are deliberate RMM/CUDA ports), so the CUDA
and ROCm modules differ only in the tables they bind here: a `GpuSymbols`
symbol spelling per entry point, an error type, and the platform's magic
values (`GpuPlatform`).

Design-for-testability (design §9): all control flow runs over an injected
`GpuApi` that wraps every native entry point. Only the concrete
`NativeGpuApi` construction needs a loadable runtime library; everything
else is exercised on CPU-only machines against a scripted fake.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import importlib
import threading
import weakref
from collections.abc import Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from types import ModuleType
from typing import Any, ClassVar, Literal, Protocol

from devmm._core.device import Device, DeviceType
from devmm._core.memory_resource import DeviceMemoryResource
from devmm._core.stream import DEFAULT, Stream, StreamError, StreamSentinel
from devmm._runtimes.base import CopyKind, RuntimeUnavailableError

# cudaSuccess and hipSuccess are both 0.
GPU_SUCCESS = 0


class GpuError(RuntimeError):
    """A native runtime call returned a nonzero status code.

    The dedicated exception types cover allocation (`MemoryError`), DLPack
    refusals (`BufferError`) and handoff failures (`StreamError`); every
    other driver failure surfaces as this `RuntimeError` (design §8).
    Platform modules subclass with their `error_label` ("CUDA", "HIP").
    """

    error_label: ClassVar[str] = "GPU"

    def __init__(self, function: str, status: int, detail: str) -> None:
        super().__init__(f"{function} failed with {self.error_label} error {status}: {detail}")
        self.function = function
        self.status = status


@dataclass(frozen=True, slots=True)
class GpuSymbols:
    """Native entry-point spellings, one per `GpuApi` method.

    Error messages and native-library prototypes both go through this
    table, so a message always names the platform's own function
    (cudaMalloc vs hipMalloc) even though the calling code is shared.
    """

    get_error_string: str
    get_device_count: str
    get_device: str
    set_device: str
    get_device_attribute: str
    malloc: str
    free: str
    malloc_async: str
    free_async: str
    stream_create: str
    stream_destroy: str
    stream_synchronize: str
    memcpy_async: str
    event_create_with_flags: str
    event_record: str
    stream_wait_event: str
    event_destroy: str


@dataclass(frozen=True, slots=True)
class GpuPlatform:
    """Everything that legitimately differs between the GPU platforms
    (design §4.2): symbol spellings, the error type, magic status/attribute
    values, sentinel stream handles and runtime-library spellings."""

    name: str
    device_type: DeviceType
    error: type[GpuError]
    symbols: GpuSymbols
    # What the async allocation entry points report when the loaded runtime
    # library predates them.
    not_supported_status: int
    # Ordering-only events, the cheap kind the DLPack handoff needs.
    event_disable_timing: int
    # The device attribute behind malloc_async/free_async support.
    memory_pools_attribute: int
    # devmm sentinel -> the platform's magic default-stream handle (§3.2).
    sentinel_handles: Mapping[StreamSentinel, int]
    # `ctypes.util.find_library` argument for the runtime library, then the
    # explicit candidate spellings tried in order.
    library_alias: str
    library_names: tuple[str, ...]
    # Actionable message when no runtime library is loadable.
    library_hint: str


class GpuApi(Protocol):
    """Every native entry point the runtimes and raw MRs touch (design §9).

    Methods mirror the C signatures, returning status codes (with
    out-parameters folded into the return tuple) instead of raising, so all
    error mapping and sequencing stays in the callers where a scripted fake
    can exercise it. `platform` binds the symbol spellings used in error
    messages and call logs.
    """

    platform: GpuPlatform

    def get_error_string(self, status: int) -> str: ...

    def get_device_count(self) -> tuple[int, int]: ...

    def get_device(self) -> tuple[int, int]: ...

    def set_device(self, index: int) -> int: ...

    def get_device_attribute(self, attribute: int, index: int) -> tuple[int, int]: ...

    def malloc(self, nbytes: int) -> tuple[int, int]: ...

    def free(self, ptr: int) -> int: ...

    def malloc_async(self, nbytes: int, stream_handle: int) -> tuple[int, int]: ...

    def free_async(self, ptr: int, stream_handle: int) -> int: ...

    def stream_create(self) -> tuple[int, int]: ...

    def stream_destroy(self, handle: int) -> int: ...

    def stream_synchronize(self, handle: int) -> int: ...

    def memcpy_async(
        self, dst: int, src: int, nbytes: int, kind: int, stream_handle: int
    ) -> int: ...

    def event_create_with_flags(self, flags: int) -> tuple[int, int]: ...

    def event_record(self, event: int, stream_handle: int) -> int: ...

    def stream_wait_event(self, stream_handle: int, event: int, flags: int) -> int: ...

    def event_destroy(self, event: int) -> int: ...


def check_status(api: GpuApi, function: str, status: int) -> None:
    if status != GPU_SUCCESS:
        raise api.platform.error(function, status, api.get_error_string(status))


def _stream_error(api: GpuApi, function: str, status: int) -> StreamError:
    label = api.platform.error.error_label
    return StreamError(
        f"{function} failed with {label} error {status}: {api.get_error_string(status)}"
    )


def event_order(api: GpuApi, *, waiter_handle: int, source_handle: int) -> None:
    """Make `waiter_handle` wait on work enqueued on `source_handle`:
    create -> record(source) -> stream-wait(waiter) -> destroy (design §7.3)."""
    symbols = api.platform.symbols
    status, event = api.event_create_with_flags(api.platform.event_disable_timing)
    if status != GPU_SUCCESS:
        raise _stream_error(api, symbols.event_create_with_flags, status)
    try:
        status = api.event_record(event, source_handle)
        if status != GPU_SUCCESS:
            raise _stream_error(api, symbols.event_record, status)
        status = api.stream_wait_event(waiter_handle, event, 0)
        if status != GPU_SUCCESS:
            raise _stream_error(api, symbols.stream_wait_event, status)
    except BaseException:
        # Best-effort cleanup: the pending error already names the root
        # cause, so a destroy failure here is not allowed to mask it.
        api.event_destroy(event)
        raise
    status = api.event_destroy(event)
    if status != GPU_SUCCESS:
        raise _stream_error(api, symbols.event_destroy, status)


@contextmanager
def device_activation(api: GpuApi, device: Device) -> Iterator[None]:
    """Flip the native active device for the scope, restoring the previous
    device even when the body raises (design §3.1); a no-op when `device` is
    already current."""
    symbols = api.platform.symbols
    status, previous = api.get_device()
    check_status(api, symbols.get_device, status)
    if previous == device.index:
        yield
        return
    check_status(api, symbols.set_device, api.set_device(device.index))
    try:
        yield
    finally:
        # Best-effort restore: raising out of a finally would mask the
        # body's exception, so a failed restore is swallowed.
        api.set_device(previous)


class GpuStream(Stream):
    """A native stream handle bound to a `GpuApi` (design §3.2).

    Wraps foreign handles as-is; streams created by
    `GpuRuntime.create_stream` additionally carry a finalizer that destroys
    the native stream when the wrapper dies.
    """

    def __init__(self, device: Device, handle: int, api: GpuApi) -> None:
        platform = api.platform
        if device.type is not platform.device_type:
            raise ValueError(
                f"{type(self).__name__} requires a {platform.name} device, got {device}"
            )
        if handle < 0:
            raise ValueError(f"stream handles are non-negative ints, got {handle}")
        self.device = device
        self._handle = handle
        self._api = api

    @property
    def handle(self) -> int:
        return self._handle

    def synchronize(self) -> None:
        api = self._api
        check_status(
            api, api.platform.symbols.stream_synchronize, api.stream_synchronize(self._handle)
        )

    def wait_raw(self, other_handle: int) -> None:
        event_order(self._api, waiter_handle=self._handle, source_handle=other_handle)


def _destroy_native_stream(api: GpuApi, handle: int) -> None:
    """`create_stream` finalizer body: a finalizer has no recovery path, so
    a failed destroy is ignored."""
    api.stream_destroy(handle)


def load_first_library(names: tuple[str, ...]) -> ctypes.CDLL | None:
    """Seam for tests: the first dlopen-able library among `names`."""
    for name in names:
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    return None


def _library_candidates(platform: GpuPlatform) -> tuple[str, ...]:
    """Candidate runtime-library spellings: the linker's own resolution
    first, then the platform's explicit list."""
    found = ctypes.util.find_library(platform.library_alias)
    return (found, *platform.library_names) if found else platform.library_names


def load_native_api(platform: GpuPlatform) -> GpuApi:
    """The production `GpuApi` over the process's runtime library — the only
    GPU-only construction path (design §9)."""
    lib = load_first_library(_library_candidates(platform))
    if lib is None:
        raise RuntimeUnavailableError(platform.library_hint)
    return NativeGpuApi(lib, platform)


class NativeGpuApi:
    """`GpuApi` over a loaded runtime library.

    ctypes prototypes are declared once here — the C signatures are
    identical on both platforms, only the symbol spellings differ; see
    `GpuApi` for why the methods return statuses instead of raising.
    """

    def __init__(self, lib: ctypes.CDLL, platform: GpuPlatform) -> None:
        self.platform = platform
        symbols = platform.symbols
        c_int = ctypes.c_int
        c_uint = ctypes.c_uint
        c_void_p = ctypes.c_void_p
        c_size_t = ctypes.c_size_t
        prototypes: dict[str, tuple[Any, list[Any]]] = {
            symbols.get_error_string: (ctypes.c_char_p, [c_int]),
            symbols.get_device_count: (c_int, [ctypes.POINTER(c_int)]),
            symbols.get_device: (c_int, [ctypes.POINTER(c_int)]),
            symbols.set_device: (c_int, [c_int]),
            symbols.get_device_attribute: (c_int, [ctypes.POINTER(c_int), c_int, c_int]),
            symbols.malloc: (c_int, [ctypes.POINTER(c_void_p), c_size_t]),
            symbols.free: (c_int, [c_void_p]),
            symbols.stream_create: (c_int, [ctypes.POINTER(c_void_p)]),
            symbols.stream_destroy: (c_int, [c_void_p]),
            symbols.stream_synchronize: (c_int, [c_void_p]),
            symbols.memcpy_async: (c_int, [c_void_p, c_void_p, c_size_t, c_int, c_void_p]),
            symbols.event_create_with_flags: (c_int, [ctypes.POINTER(c_void_p), c_uint]),
            symbols.event_record: (c_int, [c_void_p, c_void_p]),
            symbols.stream_wait_event: (c_int, [c_void_p, c_void_p, c_uint]),
            symbols.event_destroy: (c_int, [c_void_p]),
        }
        for name, (restype, argtypes) in prototypes.items():
            function = getattr(lib, name)
            function.restype = restype
            function.argtypes = argtypes
        self._c_get_error_string = getattr(lib, symbols.get_error_string)
        self._c_get_device_count = getattr(lib, symbols.get_device_count)
        self._c_get_device = getattr(lib, symbols.get_device)
        self._c_set_device = getattr(lib, symbols.set_device)
        self._c_get_device_attribute = getattr(lib, symbols.get_device_attribute)
        self._c_malloc = getattr(lib, symbols.malloc)
        self._c_free = getattr(lib, symbols.free)
        self._c_stream_create = getattr(lib, symbols.stream_create)
        self._c_stream_destroy = getattr(lib, symbols.stream_destroy)
        self._c_stream_synchronize = getattr(lib, symbols.stream_synchronize)
        self._c_memcpy_async = getattr(lib, symbols.memcpy_async)
        self._c_event_create_with_flags = getattr(lib, symbols.event_create_with_flags)
        self._c_event_record = getattr(lib, symbols.event_record)
        self._c_stream_wait_event = getattr(lib, symbols.stream_wait_event)
        self._c_event_destroy = getattr(lib, symbols.event_destroy)
        # The async allocation family arrived later than the rest (CUDA
        # 11.2, ROCm 5.2); an older runtime library simply lacks the
        # symbols and the async methods report `not_supported_status`
        # through the api instead.
        self._c_malloc_async: Any = None
        self._c_free_async: Any = None
        try:
            malloc_async = getattr(lib, symbols.malloc_async)
            free_async = getattr(lib, symbols.free_async)
        except AttributeError:
            pass
        else:
            malloc_async.restype = c_int
            malloc_async.argtypes = [ctypes.POINTER(c_void_p), c_size_t, c_void_p]
            free_async.restype = c_int
            free_async.argtypes = [c_void_p, c_void_p]
            self._c_malloc_async = malloc_async
            self._c_free_async = free_async

    def get_error_string(self, status: int) -> str:
        raw = self._c_get_error_string(status)
        return raw.decode() if raw else f"unknown {self.platform.error.error_label} error {status}"

    def get_device_count(self) -> tuple[int, int]:
        count = ctypes.c_int(0)
        return int(self._c_get_device_count(ctypes.byref(count))), count.value

    def get_device(self) -> tuple[int, int]:
        index = ctypes.c_int(0)
        return int(self._c_get_device(ctypes.byref(index))), index.value

    def set_device(self, index: int) -> int:
        return int(self._c_set_device(index))

    def get_device_attribute(self, attribute: int, index: int) -> tuple[int, int]:
        value = ctypes.c_int(0)
        status = self._c_get_device_attribute(ctypes.byref(value), attribute, index)
        return int(status), value.value

    def malloc(self, nbytes: int) -> tuple[int, int]:
        ptr = ctypes.c_void_p(None)
        status = int(self._c_malloc(ctypes.byref(ptr), nbytes))
        return status, ptr.value or 0

    def free(self, ptr: int) -> int:
        return int(self._c_free(ctypes.c_void_p(ptr)))

    def malloc_async(self, nbytes: int, stream_handle: int) -> tuple[int, int]:
        if self._c_malloc_async is None:
            return self.platform.not_supported_status, 0
        ptr = ctypes.c_void_p(None)
        status = int(
            self._c_malloc_async(ctypes.byref(ptr), nbytes, ctypes.c_void_p(stream_handle))
        )
        return status, ptr.value or 0

    def free_async(self, ptr: int, stream_handle: int) -> int:
        if self._c_free_async is None:
            return self.platform.not_supported_status
        return int(self._c_free_async(ctypes.c_void_p(ptr), ctypes.c_void_p(stream_handle)))

    def stream_create(self) -> tuple[int, int]:
        handle = ctypes.c_void_p(None)
        status = int(self._c_stream_create(ctypes.byref(handle)))
        return status, handle.value or 0

    def stream_destroy(self, handle: int) -> int:
        return int(self._c_stream_destroy(ctypes.c_void_p(handle)))

    def stream_synchronize(self, handle: int) -> int:
        return int(self._c_stream_synchronize(ctypes.c_void_p(handle)))

    def memcpy_async(self, dst: int, src: int, nbytes: int, kind: int, stream_handle: int) -> int:
        return int(
            self._c_memcpy_async(
                ctypes.c_void_p(dst),
                ctypes.c_void_p(src),
                nbytes,
                kind,
                ctypes.c_void_p(stream_handle),
            )
        )

    def event_create_with_flags(self, flags: int) -> tuple[int, int]:
        event = ctypes.c_void_p(None)
        status = int(self._c_event_create_with_flags(ctypes.byref(event), flags))
        return status, event.value or 0

    def event_record(self, event: int, stream_handle: int) -> int:
        return int(self._c_event_record(ctypes.c_void_p(event), ctypes.c_void_p(stream_handle)))

    def stream_wait_event(self, stream_handle: int, event: int, flags: int) -> int:
        return int(
            self._c_stream_wait_event(ctypes.c_void_p(stream_handle), ctypes.c_void_p(event), flags)
        )

    def event_destroy(self, event: int) -> int:
        return int(self._c_event_destroy(ctypes.c_void_p(event)))


class GpuRuntime:
    """Shared `DeviceRuntime` implementation (design §4.1) over an injected
    `GpuApi` — the process's runtime library by default.

    Platform modules subclass to bind `name`, `device_types`, `platform`,
    the stream type, and the platform's default-MR chain.
    """

    name: str
    device_types: frozenset[DeviceType]
    platform: ClassVar[GpuPlatform]
    _stream_type: ClassVar[type[GpuStream]]

    def __init__(self, api: GpuApi | None = None) -> None:
        self.api = load_native_api(self.platform) if api is None else api

    def device_count(self, device_type: DeviceType) -> int:
        if device_type is not self.platform.device_type:
            return 0
        status, count = self.api.get_device_count()
        check_status(self.api, self.platform.symbols.get_device_count, status)
        return count

    def default_memory_resource(self, device: Device) -> DeviceMemoryResource:
        raise NotImplementedError

    def default_stream(self, device: Device) -> Stream:
        self._check_device(device)
        return self._stream_type(device, self.platform.sentinel_handles[DEFAULT], self.api)

    def create_stream(self, device: Device) -> Stream:
        self._check_device(device)
        with self.activate_device(device):
            status, handle = self.api.stream_create()
        check_status(self.api, self.platform.symbols.stream_create, status)
        stream = self._stream_type(device, handle, self.api)
        weakref.finalize(stream, _destroy_native_stream, self.api, handle)
        return stream

    def wrap_stream(self, device: Device, obj: object) -> Stream:
        self._check_device(device)
        if isinstance(obj, Stream):
            if obj.device != device:
                raise ValueError(f"stream lives on {obj.device}, not on {device}")
            return obj
        if isinstance(obj, StreamSentinel):
            return self._stream_type(device, self.platform.sentinel_handles[obj], self.api)
        handle = obj
        if not isinstance(handle, int):
            cuda_stream = getattr(obj, "__cuda_stream__", None)
            if cuda_stream is None:
                raise TypeError(
                    f"cannot wrap {type(obj).__name__!r} as a stream: expected a "
                    "Stream, a sentinel, a raw int handle, or an object exposing "
                    "__cuda_stream__"
                )
            _, handle = cuda_stream()
            if not isinstance(handle, int):
                raise TypeError(f"__cuda_stream__ must yield an int handle, got {handle!r}")
        return self._stream_type(device, handle, self.api)

    def make_stream_wait(self, consumer_handle: int, producer: Stream) -> None:
        event_order(self.api, waiter_handle=consumer_handle, source_handle=producer.handle)

    def memcpy(self, dst: int, src: int, nbytes: int, kind: CopyKind, stream: Stream) -> None:
        if nbytes < 0:
            raise ValueError(f"cannot copy a negative size ({nbytes} bytes)")
        if nbytes == 0:
            return
        symbols = self.platform.symbols
        check_status(
            self.api,
            symbols.memcpy_async,
            self.api.memcpy_async(dst, src, nbytes, int(kind), stream.handle),
        )
        if kind is not CopyKind.DEVICE_TO_DEVICE:
            # Host pointers in the transfer are not stream-ordered: the
            # caller typically stages through a temporary host buffer that
            # may die the moment this call returns, so host-involving
            # copies complete before returning. Device-to-device stays
            # stream-ordered.
            check_status(
                self.api,
                symbols.stream_synchronize,
                self.api.stream_synchronize(stream.handle),
            )

    def activate_device(self, device: Device) -> AbstractContextManager[None]:
        self._check_device(device)
        return device_activation(self.api, device)

    def _check_device(self, device: Device) -> None:
        if device.type is not self.platform.device_type:
            raise ValueError(
                f"{type(self).__name__} serves {self.platform.name} devices, got {device}"
            )


class GpuRuntimeMemoryResource(DeviceMemoryResource):
    """The "just malloc/free on the runtime library" MR (design §5.2, §5.3),
    no third-party dependency; platform modules subclass to bind the table.

    `async_alloc="auto"` probes the platform's memory-pools attribute once
    and picks the malloc_async/free_async family iff the driver supports
    memory pools (a failed probe means "not supported": older drivers
    predate the attribute); `True`/`False` force the async/sync family
    without probing. `stream_ordered` is True only on the async path — the
    sync path relies on the platform free's implicit synchronization for
    safety. Both families return pointers aligned to at least 256 bytes.
    """

    platform: ClassVar[GpuPlatform]

    def __init__(
        self,
        device: Device,
        *,
        async_alloc: bool | Literal["auto"] = "auto",
        api: GpuApi | None = None,
    ) -> None:
        platform = self.platform
        if device.type is not platform.device_type:
            raise ValueError(
                f"{type(self).__name__} requires a {platform.name} device, got {device}"
            )
        self.device = device
        self._api = load_native_api(platform) if api is None else api
        if async_alloc == "auto":
            status, supported = self._api.get_device_attribute(
                platform.memory_pools_attribute, device.index
            )
            self._async = status == GPU_SUCCESS and bool(supported)
        elif async_alloc is True or async_alloc is False:
            self._async = async_alloc
        else:
            raise ValueError(f"async_alloc must be 'auto', True or False, got {async_alloc!r}")
        self._lock = threading.Lock()
        # ptr -> nbytes: deallocate validates the pointer and size before any
        # driver call, so misuse surfaces as ValueError, never memory
        # corruption.
        self._live: dict[int, int] = {}

    def allocate(self, nbytes: int, stream: Stream) -> int:
        if nbytes < 0:
            raise ValueError(f"cannot allocate a negative size ({nbytes} bytes)")
        # max(nbytes, 1): a zero-byte malloc returns NULL, which could be
        # neither tracked nor freed; one byte buys a unique, freeable
        # pointer.
        request = max(nbytes, 1)
        with device_activation(self._api, self.device):
            if self._async:
                status, ptr = self._api.malloc_async(request, stream.handle)
            else:
                status, ptr = self._api.malloc(request)
        if status != GPU_SUCCESS:
            raise MemoryError(
                f"failed to allocate {nbytes} bytes on {self.device} in {self!r}: "
                f"{self._api.get_error_string(status)}"
            )
        with self._lock:
            self._live[ptr] = nbytes
        return ptr

    def deallocate(self, ptr: int, nbytes: int, stream: Stream) -> None:
        with self._lock:
            recorded = self._live.get(ptr)
            if recorded is None:
                raise ValueError(
                    f"pointer {ptr:#x} is not a live allocation of {self!r} "
                    "(double-free, or a pointer this MR never returned)"
                )
            if nbytes != recorded:
                # The allocation stays live: a size-mismatched free is caller
                # confusion, and releasing anyway would turn it into a
                # use-after-free elsewhere.
                raise ValueError(
                    f"size mismatch freeing pointer {ptr:#x} in {self!r}: "
                    f"allocated {recorded} bytes, deallocate got {nbytes}"
                )
            del self._live[ptr]
        symbols = self.platform.symbols
        with device_activation(self._api, self.device):
            if self._async:
                status = self._api.free_async(ptr, stream.handle)
                function = symbols.free_async
            else:
                status = self._api.free(ptr)
                function = symbols.free
        check_status(self._api, function, status)

    @property
    def stream_ordered(self) -> bool:
        return self._async

    def guaranteed_alignment(self) -> int:
        # Both platforms guarantee at least 256-byte alignment from both
        # allocation families.
        return 256

    def _debug_live_count(self) -> int:
        """Testing hook: allocations currently tracked (0 == tables empty)."""
        with self._lock:
            return len(self._live)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(device={self.device}, async_alloc={self._async})"


def rmm_stream_class(unavailable_hint: str) -> Any:
    """The rmm-module Stream class, imported on demand.

    hipMM's port keeps the `rmm.pylibrmm.stream` module path verbatim, so
    both platforms' wrappers translate through the same class; the
    constructor accepts any object exposing `__cuda_stream__` — which every
    devmm `Stream` does — so wrapping the devmm stream in it is the whole
    translation (design §5.2, §5.3).
    """
    try:
        return importlib.import_module("rmm.pylibrmm.stream").Stream
    except ImportError as exc:
        raise RuntimeUnavailableError(unavailable_hint) from exc


class RmmResourceLike(Protocol):
    """The rmm/hipMM `DeviceMemoryResource` Python surface the wrappers
    forward to — a Protocol so the zero-dependency core never imports the
    module (design §8)."""

    def allocate(self, nbytes: int, stream: Any) -> int: ...

    def deallocate(self, ptr: int, nbytes: int, stream: Any) -> None: ...


class RmmLikeMemoryResource(DeviceMemoryResource):
    """Shared shape of the rmm-module wrappers (design §5.2, §5.3): hipMM's
    Python layer is a direct RMM port, so one forwarding implementation
    serves both; platform modules bind the table and the stream translation.

    `inner` is a strong reference by design — the rmm lifetime lesson
    (design §3.3): the wrapped resource must outlive every allocation made
    through it. Allocation failures raised by the wrapped module (already
    `MemoryError`s) propagate untouched. rmm-style resources allocate
    stream-ordered, 256-byte-aligned memory. The caller pairs `inner` with
    the device it was created for — the wrapped resources do not expose it.
    """

    platform: ClassVar[GpuPlatform]
    inner: RmmResourceLike

    def __init__(self, inner: RmmResourceLike, device: Device) -> None:
        platform = self.platform
        if device.type is not platform.device_type:
            raise ValueError(
                f"{type(self).__name__} requires a {platform.name} device, got {device}"
            )
        self.inner = inner
        self.device = device
        self._lock = threading.Lock()
        # ptr -> nbytes, for the same misuse detection the sibling MRs give.
        self._live: dict[int, int] = {}

    def _translated_stream(self, stream: Stream) -> Any:
        """The rmm-module stream object for `stream` (platform-bound seam)."""
        raise NotImplementedError

    def allocate(self, nbytes: int, stream: Stream) -> int:
        if nbytes < 0:
            raise ValueError(f"cannot allocate a negative size ({nbytes} bytes)")
        # max(nbytes, 1): rmm forwards 0 to allocators that may return NULL
        # or a shared address; one byte buys a unique, freeable pointer.
        ptr = int(self.inner.allocate(max(nbytes, 1), self._translated_stream(stream)))
        with self._lock:
            self._live[ptr] = nbytes
        return ptr

    def deallocate(self, ptr: int, nbytes: int, stream: Stream) -> None:
        with self._lock:
            recorded = self._live.get(ptr)
            if recorded is None:
                raise ValueError(
                    f"pointer {ptr:#x} is not a live allocation of {self!r} "
                    "(double-free, or a pointer this MR never returned)"
                )
            if nbytes != recorded:
                raise ValueError(
                    f"size mismatch freeing pointer {ptr:#x} in {self!r}: "
                    f"allocated {recorded} bytes, deallocate got {nbytes}"
                )
            del self._live[ptr]
        self.inner.deallocate(ptr, max(nbytes, 1), self._translated_stream(stream))

    @property
    def stream_ordered(self) -> bool:
        return True

    def guaranteed_alignment(self) -> int:
        # The rmm contract, kept by the hipMM port: allocations are aligned
        # to at least 256 bytes.
        return 256

    def _debug_live_count(self) -> int:
        """Testing hook: allocations currently tracked (0 == tables empty)."""
        with self._lock:
            return len(self._live)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(inner={self.inner!r}, device={self.device})"


# hipMM's Python package installs a module also named `rmm` (a straight
# RAPIDS port), so the module name identifies nothing; the concrete
# `rmm.mr` resource-class surface does (design §4.2): RMM ships Cuda*-named
# resources, hipMM ships Hip*-named ones.
_RMM_MARKERS: dict[str, tuple[str, ...]] = {
    "cuda": ("CudaMemoryResource", "CudaAsyncMemoryResource"),
    "rocm": ("HipMemoryResource", "HipAsyncMemoryResource"),
}


def rmm_module_platform(module: ModuleType) -> str | None:
    """Best-effort platform of an `rmm`-named module: a runtime name, or
    None when the marker surface is missing or ambiguous."""
    mr_namespace = getattr(module, "mr", None)
    if mr_namespace is None:
        return None
    matches = [
        name
        for name, markers in _RMM_MARKERS.items()
        if any(hasattr(mr_namespace, marker) for marker in markers)
    ]
    if len(matches) == 1:
        return matches[0]
    return None


def import_rmm_for(platform_name: str) -> ModuleType | None:
    """The `rmm` module iff it imports and verifiably targets
    `platform_name`; None otherwise, so the caller falls back to the raw
    runtime MR (design §4.2). `DEVMM_RUNTIME` remains the explicit override
    for pathological environments — it selects the runtime, and each
    runtime only ever adopts a platform-matching `rmm`."""
    try:
        module = importlib.import_module("rmm")
    except ImportError:
        return None
    if rmm_module_platform(module) != platform_name:
        return None
    return module
