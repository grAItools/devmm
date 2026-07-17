"""CPU runtime: single no-op stream, host memcpy, MallocMR default (§4)."""

from __future__ import annotations

import ctypes
from contextlib import AbstractContextManager, nullcontext

from devmm._core.device import Device, DeviceType
from devmm._core.memory_resource import DeviceMemoryResource
from devmm._core.stream import CpuStream, Stream
from devmm._runtimes.base import CopyKind
from devmm.mrs.cpu import MallocMemoryResource


class CpuRuntime:
    """The host platform's `DeviceRuntime` (design §4.1).

    Host work is synchronous: every stream is the single no-op `CpuStream`,
    the ordering primitives do nothing, and `memcpy` is `ctypes.memmove`.
    The default MR is `MallocMemoryResource` (design §4.1's CPU chain).
    """

    name = "cpu"
    device_types = frozenset({DeviceType.CPU})

    def device_count(self, device_type: DeviceType) -> int:
        # The host is a single device; index 0 is the only ordinal.
        return 1 if device_type is DeviceType.CPU else 0

    def default_memory_resource(self, device: Device) -> DeviceMemoryResource:
        self._check_device(device)
        return MallocMemoryResource(device)

    def default_stream(self, device: Device) -> Stream:
        self._check_device(device)
        return CpuStream(device)

    def create_stream(self, device: Device) -> Stream:
        self._check_device(device)
        return CpuStream(device)

    def wrap_stream(self, device: Device, obj: object) -> Stream:
        self._check_device(device)
        if isinstance(obj, Stream):
            if obj.device != device:
                raise ValueError(f"stream lives on {obj.device}, not on {device}")
            return obj
        handle = obj
        if not isinstance(handle, int):
            cuda_stream = getattr(obj, "__cuda_stream__", None)
            if cuda_stream is None:
                raise TypeError(
                    f"cannot wrap {type(obj).__name__!r} as a stream: expected a "
                    "Stream, a raw int handle, or an object exposing __cuda_stream__"
                )
            _, handle = cuda_stream()
        if handle != 0:
            raise ValueError(f"the CPU has a single stream with handle 0, got handle {handle}")
        return CpuStream(device)

    def make_stream_wait(self, consumer_handle: int, producer: Stream) -> None:
        # Host work completes synchronously: anything enqueued on the (only)
        # CPU stream is already done, so there is nothing to wait for.
        return None

    def memcpy(self, dst: int, src: int, nbytes: int, kind: CopyKind, stream: Stream) -> None:
        if kind is not CopyKind.HOST_TO_HOST:
            raise ValueError(f"the CPU runtime only copies host-to-host, got {kind.name}")
        if nbytes < 0:
            raise ValueError(f"cannot copy a negative size ({nbytes} bytes)")
        if nbytes:
            ctypes.memmove(dst, src, nbytes)

    def activate_device(self, device: Device) -> AbstractContextManager[None]:
        self._check_device(device)
        # There is no native active-device state to flip on the host.
        return nullcontext()

    def _check_device(self, device: Device) -> None:
        if device.type is not DeviceType.CPU:
            raise ValueError(f"{type(self).__name__} serves cpu devices, got {device}")
