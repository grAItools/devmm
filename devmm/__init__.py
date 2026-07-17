"""devmm — Device Memory Manager.

A uniform, pure-Python interface for allocating and managing device memory
across CPU/CUDA/ROCm, exposing allocations as DLPack >= 1.0 producers.
See work/devmm-design.md for the full architecture.

Public re-exports (empty, empty_like, Device, Stream, Tensor, DeviceBuffer,
Layout, LayoutPolicy, DeviceMemoryResource, available_runtimes, ...) are
added here as the core modules are implemented.
"""

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

__version__ = "0.1.0"

__all__ = [
    "Aligned",
    "ColMajor",
    "DType",
    "Device",
    "DeviceOptimal",
    "DeviceType",
    "Layout",
    "LayoutPolicy",
    "Permuted",
    "RowMajor",
]
