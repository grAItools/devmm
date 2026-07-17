"""devmm — Device Memory Manager.

A uniform, pure-Python interface for allocating and managing device memory
across CPU/CUDA/ROCm, exposing allocations as DLPack >= 1.0 producers.
See work/devmm-design.md for the full architecture.

This module holds the public re-exports only (design §2): the factories
(`empty`, `empty_like`), the domain model (`Device`, `Stream`, `Tensor`,
`DeviceBuffer`, `Layout`, `DType`, memory resources and adaptors), the
current-MR registry accessors, and the runtime query API
(`available_runtimes`, `runtime_names`, `runtime_for`).
"""

from devmm._core.buffer import DeviceBuffer
from devmm._core.device import Device, DeviceType
from devmm._core.dtypes import DType
from devmm._core.layout import (
    Aligned,
    ColMajor,
    DeviceOptimal,
    Layout,
    LayoutPolicy,
    Permuted,
    RowMajor,
)
from devmm._core.memory_resource import (
    CallbackMemoryResource,
    DeviceMemoryResource,
    LimitingAdaptor,
    LoggingAdaptor,
    StatisticsAdaptor,
)
from devmm._core.registry import (
    get_current_memory_resource,
    set_current_memory_resource,
    using_memory_resource,
)
from devmm._core.stream import DEFAULT, LEGACY_DEFAULT, PER_THREAD_DEFAULT, Stream
from devmm._core.tensor import Tensor, empty, empty_like
from devmm._runtimes._discovery import available_runtimes, runtime_for, runtime_names

__version__ = "0.1.0"

__all__ = [
    "DEFAULT",
    "LEGACY_DEFAULT",
    "PER_THREAD_DEFAULT",
    "Aligned",
    "CallbackMemoryResource",
    "ColMajor",
    "DType",
    "Device",
    "DeviceBuffer",
    "DeviceMemoryResource",
    "DeviceOptimal",
    "DeviceType",
    "Layout",
    "LayoutPolicy",
    "LimitingAdaptor",
    "LoggingAdaptor",
    "Permuted",
    "RowMajor",
    "StatisticsAdaptor",
    "Stream",
    "Tensor",
    "available_runtimes",
    "empty",
    "empty_like",
    "get_current_memory_resource",
    "runtime_for",
    "runtime_names",
    "set_current_memory_resource",
    "using_memory_resource",
]
