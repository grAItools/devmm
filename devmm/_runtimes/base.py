"""The `DeviceRuntime` SPI (Protocol): device enumeration, streams/events, memcpy, native device
activation, and default-MR policy (§4.1).
"""

from __future__ import annotations

import enum
from contextlib import AbstractContextManager
from typing import Protocol

from devmm._core.device import Device, DeviceType
from devmm._core.memory_resource import DeviceMemoryResource
from devmm._core.stream import Stream


class RuntimeUnavailableError(ImportError):
    """No device runtime can serve the request (design §8).

    Subclasses `ImportError` because the root cause is a missing or
    unloadable platform stack (driver library, runtime module).
    """


class CopyKind(enum.IntEnum):
    """Direction of a `DeviceRuntime.memcpy`.

    Values are the CUDA/HIP `cudaMemcpyKind`/`hipMemcpyKind` codes verbatim,
    so runtime FFI shims never need a translation table (the same trick
    `DeviceType` plays with DLPack codes, design §3.1). `DEFAULT` is the
    UVA "infer the direction from the pointers" kind.
    """

    HOST_TO_HOST = 0
    HOST_TO_DEVICE = 1
    DEVICE_TO_HOST = 2
    DEVICE_TO_DEVICE = 3
    DEFAULT = 4


class DeviceRuntime(Protocol):
    """The per-platform runtime SPI (design §4.1).

    Exactly one exists per platform: it owns device enumeration, stream
    factories, memcpy, native device activation and the default-MR policy.
    It never allocates memory itself — that is what `DeviceMemoryResource`s
    are for. There is deliberately no `is_available()`: availability is a
    property of the environment, answered by the discovery probe *before*
    construction; an instantiated runtime is available by construction.
    """

    name: str
    device_types: frozenset[DeviceType]

    def device_count(self, device_type: DeviceType) -> int:
        """Number of devices of `device_type` this runtime can see."""
        ...

    def default_memory_resource(self, device: Device) -> DeviceMemoryResource:
        """The MR the registry adopts when none was set for `device` (§3.4)."""
        ...

    def default_stream(self, device: Device) -> Stream:
        """The platform's default stream for `device`."""
        ...

    def create_stream(self, device: Device) -> Stream:
        """A new native stream on `device`."""
        ...

    def wrap_stream(self, device: Device, obj: object) -> Stream:
        """Adopt a foreign stream: a raw int handle, an object exposing
        `__cuda_stream__`, or an existing `Stream` on `device`."""
        ...

    def make_stream_wait(self, consumer_handle: int, producer: Stream) -> None:
        """Make the raw consumer stream wait on work enqueued on `producer`
        (event record + stream-wait-event); the `__dlpack__(stream=...)`
        handoff primitive (design §7.3)."""
        ...

    def memcpy(self, dst: int, src: int, nbytes: int, kind: CopyKind, stream: Stream) -> None:
        """Copy `nbytes` from pointer `src` to pointer `dst`, ordered on
        `stream` where the platform supports it."""
        ...

    def activate_device(self, device: Device) -> AbstractContextManager[None]:
        """Flip the runtime's native active device for the scope (§3.1)."""
        ...
