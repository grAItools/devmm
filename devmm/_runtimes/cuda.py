"""CUDA runtime over libcudart via ctypes; rmm-or-cudaMalloc default MR (§4).

Design-for-testability (design §9): the runtime — and the raw MR in
`devmm.mrs.cuda` — run all control flow over an injected `CudartApi` that
wraps every libcudart entry point. Only the concrete `_LibcudartApi`
construction needs a loadable libcudart; everything else is exercised on
CPU-only machines against a scripted fake.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import importlib
import sys
import weakref
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from types import ModuleType
from typing import Any, Protocol

from devmm._core.device import Device, DeviceType
from devmm._core.memory_resource import DeviceMemoryResource
from devmm._core.stream import (
    DEFAULT,
    LEGACY_DEFAULT,
    PER_THREAD_DEFAULT,
    Stream,
    StreamError,
    StreamSentinel,
)
from devmm._runtimes.base import CopyKind, RuntimeUnavailableError

CUDA_SUCCESS = 0
# cudaErrorNotSupported: what the async entry points report when the loaded
# libcudart predates them (CUDA < 11.2).
CUDA_ERROR_NOT_SUPPORTED = 801
# cudaEventDisableTiming: ordering-only events, the cheap kind the DLPack
# handoff needs.
CUDA_EVENT_DISABLE_TIMING = 0x2
# cudaDevAttrMemoryPoolsSupported: the driver capability behind
# cudaMallocAsync/cudaFreeAsync.
CUDA_DEV_ATTR_MEMORY_POOLS_SUPPORTED = 115


class CudartApi(Protocol):
    """Every libcudart entry point the runtime and raw MR touch (design §9).

    Methods mirror the C signatures, returning `cudaError_t` statuses (with
    out-parameters folded into the return tuple) instead of raising, so all
    error mapping and sequencing stays in the callers where a scripted fake
    can exercise it.
    """

    def cudaGetErrorString(self, status: int) -> str: ...

    def cudaGetDeviceCount(self) -> tuple[int, int]: ...

    def cudaGetDevice(self) -> tuple[int, int]: ...

    def cudaSetDevice(self, index: int) -> int: ...

    def cudaDeviceGetAttribute(self, attribute: int, index: int) -> tuple[int, int]: ...

    def cudaMalloc(self, nbytes: int) -> tuple[int, int]: ...

    def cudaFree(self, ptr: int) -> int: ...

    def cudaMallocAsync(self, nbytes: int, stream_handle: int) -> tuple[int, int]: ...

    def cudaFreeAsync(self, ptr: int, stream_handle: int) -> int: ...

    def cudaStreamCreate(self) -> tuple[int, int]: ...

    def cudaStreamDestroy(self, handle: int) -> int: ...

    def cudaStreamSynchronize(self, handle: int) -> int: ...

    def cudaMemcpyAsync(
        self, dst: int, src: int, nbytes: int, kind: int, stream_handle: int
    ) -> int: ...

    def cudaEventCreateWithFlags(self, flags: int) -> tuple[int, int]: ...

    def cudaEventRecord(self, event: int, stream_handle: int) -> int: ...

    def cudaStreamWaitEvent(self, stream_handle: int, event: int, flags: int) -> int: ...

    def cudaEventDestroy(self, event: int) -> int: ...


class CudaError(RuntimeError):
    """A libcudart call returned a nonzero `cudaError_t`.

    The dedicated exception types cover allocation (`MemoryError`), DLPack
    refusals (`BufferError`) and handoff failures (`StreamError`); every
    other driver failure surfaces as this `RuntimeError` (design §8).
    """

    def __init__(self, function: str, status: int, detail: str) -> None:
        super().__init__(f"{function} failed with CUDA error {status}: {detail}")
        self.function = function
        self.status = status


def _check(api: CudartApi, function: str, status: int) -> None:
    if status != CUDA_SUCCESS:
        raise CudaError(function, status, api.cudaGetErrorString(status))


def _stream_error(api: CudartApi, function: str, status: int) -> StreamError:
    return StreamError(
        f"{function} failed with CUDA error {status}: {api.cudaGetErrorString(status)}"
    )


def _event_order(api: CudartApi, *, waiter_handle: int, source_handle: int) -> None:
    """Make `waiter_handle` wait on work enqueued on `source_handle`:
    create -> record(source) -> stream-wait(waiter) -> destroy (design §7.3)."""
    status, event = api.cudaEventCreateWithFlags(CUDA_EVENT_DISABLE_TIMING)
    if status != CUDA_SUCCESS:
        raise _stream_error(api, "cudaEventCreateWithFlags", status)
    try:
        status = api.cudaEventRecord(event, source_handle)
        if status != CUDA_SUCCESS:
            raise _stream_error(api, "cudaEventRecord", status)
        status = api.cudaStreamWaitEvent(waiter_handle, event, 0)
        if status != CUDA_SUCCESS:
            raise _stream_error(api, "cudaStreamWaitEvent", status)
    except BaseException:
        # Best-effort cleanup: the pending error already names the root
        # cause, so a destroy failure here is not allowed to mask it.
        api.cudaEventDestroy(event)
        raise
    status = api.cudaEventDestroy(event)
    if status != CUDA_SUCCESS:
        raise _stream_error(api, "cudaEventDestroy", status)


@contextmanager
def _device_activation(api: CudartApi, device: Device) -> Iterator[None]:
    """Flip the native active device for the scope, restoring the previous
    device even when the body raises (design §3.1); a no-op when `device` is
    already current."""
    status, previous = api.cudaGetDevice()
    _check(api, "cudaGetDevice", status)
    if previous == device.index:
        yield
        return
    _check(api, "cudaSetDevice", api.cudaSetDevice(device.index))
    try:
        yield
    finally:
        # Best-effort restore: raising out of a finally would mask the
        # body's exception, so a failed restore is swallowed.
        api.cudaSetDevice(previous)


class CudaStream(Stream):
    """A CUDA stream handle bound to a `CudartApi` (design §3.2).

    Wraps foreign handles as-is; streams created by
    `CudaRuntime.create_stream` additionally carry a finalizer that destroys
    the native stream when the wrapper dies.
    """

    def __init__(self, device: Device, handle: int, api: CudartApi) -> None:
        if device.type is not DeviceType.CUDA:
            raise ValueError(f"CudaStream requires a cuda device, got {device}")
        if handle < 0:
            raise ValueError(f"stream handles are non-negative ints, got {handle}")
        self.device = device
        self._handle = handle
        self._api = api

    @property
    def handle(self) -> int:
        return self._handle

    def synchronize(self) -> None:
        _check(self._api, "cudaStreamSynchronize", self._api.cudaStreamSynchronize(self._handle))

    def wait_raw(self, other_handle: int) -> None:
        _event_order(self._api, waiter_handle=self._handle, source_handle=other_handle)


def _destroy_native_stream(api: CudartApi, handle: int) -> None:
    """`create_stream` finalizer body: a finalizer has no recovery path, so
    a failed destroy is ignored."""
    api.cudaStreamDestroy(handle)


# The platform's magic default-stream handles, mirroring rmm's sentinels
# (design §3.2): cudaStreamDefault / cudaStreamLegacy / cudaStreamPerThread.
_SENTINEL_HANDLES: dict[StreamSentinel, int] = {
    DEFAULT: 0x0,
    LEGACY_DEFAULT: 0x1,
    PER_THREAD_DEFAULT: 0x2,
}


def _import_rmm() -> ModuleType | None:
    """Seam for tests: the `rmm` module when importable, else None."""
    try:
        return importlib.import_module("rmm")
    except ImportError:
        return None


class CudaRuntime:
    """The NVIDIA platform's `DeviceRuntime` (design §4.1) over an injected
    `CudartApi` — the process's libcudart by default.

    Default-MR chain (design §4.1): `RmmMemoryResource` over rmm's
    per-device resource when rmm imports, else `CudaRuntimeMemoryResource`
    (async variant when the driver supports memory pools).
    """

    name = "cuda"
    device_types = frozenset({DeviceType.CUDA})

    def __init__(self, api: CudartApi | None = None) -> None:
        self.api = _default_api() if api is None else api

    def device_count(self, device_type: DeviceType) -> int:
        if device_type not in self.device_types:
            return 0
        status, count = self.api.cudaGetDeviceCount()
        _check(self.api, "cudaGetDeviceCount", status)
        return count

    def default_memory_resource(self, device: Device) -> DeviceMemoryResource:
        self._check_device(device)
        # Imported lazily: devmm.mrs.cuda imports this module for the api
        # machinery, so a module-scope import would be circular.
        from devmm.mrs.cuda import CudaRuntimeMemoryResource, RmmMemoryResource

        rmm = _import_rmm()
        if rmm is not None:
            # Users who already configured rmm (pools, reinitialize, ...)
            # get that configuration for free (design §5.2).
            return RmmMemoryResource(rmm.mr.get_per_device_resource(device.index), device)
        return CudaRuntimeMemoryResource(device, async_alloc="auto", api=self.api)

    def default_stream(self, device: Device) -> Stream:
        self._check_device(device)
        return CudaStream(device, _SENTINEL_HANDLES[DEFAULT], self.api)

    def create_stream(self, device: Device) -> Stream:
        self._check_device(device)
        with self.activate_device(device):
            status, handle = self.api.cudaStreamCreate()
        _check(self.api, "cudaStreamCreate", status)
        stream = CudaStream(device, handle, self.api)
        weakref.finalize(stream, _destroy_native_stream, self.api, handle)
        return stream

    def wrap_stream(self, device: Device, obj: object) -> Stream:
        self._check_device(device)
        if isinstance(obj, Stream):
            if obj.device != device:
                raise ValueError(f"stream lives on {obj.device}, not on {device}")
            return obj
        if isinstance(obj, StreamSentinel):
            return CudaStream(device, _SENTINEL_HANDLES[obj], self.api)
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
        return CudaStream(device, handle, self.api)

    def make_stream_wait(self, consumer_handle: int, producer: Stream) -> None:
        _event_order(self.api, waiter_handle=consumer_handle, source_handle=producer.handle)

    def memcpy(self, dst: int, src: int, nbytes: int, kind: CopyKind, stream: Stream) -> None:
        if nbytes < 0:
            raise ValueError(f"cannot copy a negative size ({nbytes} bytes)")
        if nbytes == 0:
            return
        _check(
            self.api,
            "cudaMemcpyAsync",
            self.api.cudaMemcpyAsync(dst, src, nbytes, int(kind), stream.handle),
        )
        if kind is not CopyKind.DEVICE_TO_DEVICE:
            # Host pointers in the transfer are not stream-ordered: the
            # caller typically stages through a temporary host buffer that
            # may die the moment this call returns, so host-involving
            # copies complete before returning. Device-to-device stays
            # stream-ordered.
            _check(
                self.api,
                "cudaStreamSynchronize",
                self.api.cudaStreamSynchronize(stream.handle),
            )

    def activate_device(self, device: Device) -> AbstractContextManager[None]:
        self._check_device(device)
        return _device_activation(self.api, device)

    def _check_device(self, device: Device) -> None:
        if device.type not in self.device_types:
            raise ValueError(f"{type(self).__name__} serves cuda devices, got {device}")


def _libcudart_names() -> tuple[str, ...]:
    """Candidate libcudart spellings: the linker's own resolution first,
    then versioned names newest-first, then the unversioned dev-symlink
    spelling (present only where the toolkit's dev files are installed)."""
    if sys.platform == "win32":
        candidates = ("cudart64_13.dll", "cudart64_12.dll", "cudart64_110.dll")
    else:
        candidates = ("libcudart.so.13", "libcudart.so.12", "libcudart.so.11", "libcudart.so")
    found = ctypes.util.find_library("cudart")
    return (found, *candidates) if found else candidates


def _load_first_library(names: tuple[str, ...]) -> ctypes.CDLL | None:
    """Seam for tests: the first dlopen-able library among `names`."""
    for name in names:
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    return None


def _default_api() -> CudartApi:
    """The production `CudartApi` over the process's libcudart — the only
    GPU-only construction path (design §9)."""
    lib = _load_first_library(_libcudart_names())
    if lib is None:
        raise RuntimeUnavailableError(
            "libcudart is not loadable; install the CUDA toolkit runtime "
            "(or a wheel bundling it) to use the cuda runtime"
        )
    return _LibcudartApi(lib)


class _LibcudartApi:
    """`CudartApi` over a loaded libcudart.

    ctypes prototypes are declared once here; see `CudartApi` for why the
    methods return statuses instead of raising.
    """

    def __init__(self, lib: ctypes.CDLL) -> None:
        self._lib = lib
        c_int = ctypes.c_int
        c_uint = ctypes.c_uint
        c_void_p = ctypes.c_void_p
        c_size_t = ctypes.c_size_t
        prototypes: dict[str, tuple[Any, list[Any]]] = {
            "cudaGetErrorString": (ctypes.c_char_p, [c_int]),
            "cudaGetDeviceCount": (c_int, [ctypes.POINTER(c_int)]),
            "cudaGetDevice": (c_int, [ctypes.POINTER(c_int)]),
            "cudaSetDevice": (c_int, [c_int]),
            "cudaDeviceGetAttribute": (c_int, [ctypes.POINTER(c_int), c_int, c_int]),
            "cudaMalloc": (c_int, [ctypes.POINTER(c_void_p), c_size_t]),
            "cudaFree": (c_int, [c_void_p]),
            "cudaStreamCreate": (c_int, [ctypes.POINTER(c_void_p)]),
            "cudaStreamDestroy": (c_int, [c_void_p]),
            "cudaStreamSynchronize": (c_int, [c_void_p]),
            "cudaMemcpyAsync": (c_int, [c_void_p, c_void_p, c_size_t, c_int, c_void_p]),
            "cudaEventCreateWithFlags": (c_int, [ctypes.POINTER(c_void_p), c_uint]),
            "cudaEventRecord": (c_int, [c_void_p, c_void_p]),
            "cudaStreamWaitEvent": (c_int, [c_void_p, c_void_p, c_uint]),
            "cudaEventDestroy": (c_int, [c_void_p]),
        }
        for name, (restype, argtypes) in prototypes.items():
            function = getattr(lib, name)
            function.restype = restype
            function.argtypes = argtypes
        # cudaMallocAsync/cudaFreeAsync arrived with CUDA 11.2; an older
        # libcudart simply lacks the symbols and the async family reports
        # cudaErrorNotSupported through the api instead.
        self._malloc_async: Any = None
        self._free_async: Any = None
        try:
            malloc_async = lib.cudaMallocAsync
            free_async = lib.cudaFreeAsync
        except AttributeError:
            pass
        else:
            malloc_async.restype = c_int
            malloc_async.argtypes = [ctypes.POINTER(c_void_p), c_size_t, c_void_p]
            free_async.restype = c_int
            free_async.argtypes = [c_void_p, c_void_p]
            self._malloc_async = malloc_async
            self._free_async = free_async

    def cudaGetErrorString(self, status: int) -> str:
        raw = self._lib.cudaGetErrorString(status)
        return raw.decode() if raw else f"unknown CUDA error {status}"

    def cudaGetDeviceCount(self) -> tuple[int, int]:
        count = ctypes.c_int(0)
        return int(self._lib.cudaGetDeviceCount(ctypes.byref(count))), count.value

    def cudaGetDevice(self) -> tuple[int, int]:
        index = ctypes.c_int(0)
        return int(self._lib.cudaGetDevice(ctypes.byref(index))), index.value

    def cudaSetDevice(self, index: int) -> int:
        return int(self._lib.cudaSetDevice(index))

    def cudaDeviceGetAttribute(self, attribute: int, index: int) -> tuple[int, int]:
        value = ctypes.c_int(0)
        status = self._lib.cudaDeviceGetAttribute(ctypes.byref(value), attribute, index)
        return int(status), value.value

    def cudaMalloc(self, nbytes: int) -> tuple[int, int]:
        ptr = ctypes.c_void_p(None)
        status = int(self._lib.cudaMalloc(ctypes.byref(ptr), nbytes))
        return status, ptr.value or 0

    def cudaFree(self, ptr: int) -> int:
        return int(self._lib.cudaFree(ctypes.c_void_p(ptr)))

    def cudaMallocAsync(self, nbytes: int, stream_handle: int) -> tuple[int, int]:
        if self._malloc_async is None:
            return CUDA_ERROR_NOT_SUPPORTED, 0
        ptr = ctypes.c_void_p(None)
        status = int(self._malloc_async(ctypes.byref(ptr), nbytes, ctypes.c_void_p(stream_handle)))
        return status, ptr.value or 0

    def cudaFreeAsync(self, ptr: int, stream_handle: int) -> int:
        if self._free_async is None:
            return CUDA_ERROR_NOT_SUPPORTED
        return int(self._free_async(ctypes.c_void_p(ptr), ctypes.c_void_p(stream_handle)))

    def cudaStreamCreate(self) -> tuple[int, int]:
        handle = ctypes.c_void_p(None)
        status = int(self._lib.cudaStreamCreate(ctypes.byref(handle)))
        return status, handle.value or 0

    def cudaStreamDestroy(self, handle: int) -> int:
        return int(self._lib.cudaStreamDestroy(ctypes.c_void_p(handle)))

    def cudaStreamSynchronize(self, handle: int) -> int:
        return int(self._lib.cudaStreamSynchronize(ctypes.c_void_p(handle)))

    def cudaMemcpyAsync(
        self, dst: int, src: int, nbytes: int, kind: int, stream_handle: int
    ) -> int:
        return int(
            self._lib.cudaMemcpyAsync(
                ctypes.c_void_p(dst),
                ctypes.c_void_p(src),
                nbytes,
                kind,
                ctypes.c_void_p(stream_handle),
            )
        )

    def cudaEventCreateWithFlags(self, flags: int) -> tuple[int, int]:
        event = ctypes.c_void_p(None)
        status = int(self._lib.cudaEventCreateWithFlags(ctypes.byref(event), flags))
        return status, event.value or 0

    def cudaEventRecord(self, event: int, stream_handle: int) -> int:
        return int(
            self._lib.cudaEventRecord(ctypes.c_void_p(event), ctypes.c_void_p(stream_handle))
        )

    def cudaStreamWaitEvent(self, stream_handle: int, event: int, flags: int) -> int:
        return int(
            self._lib.cudaStreamWaitEvent(
                ctypes.c_void_p(stream_handle), ctypes.c_void_p(event), flags
            )
        )

    def cudaEventDestroy(self, event: int) -> int:
        return int(self._lib.cudaEventDestroy(ctypes.c_void_p(event)))
