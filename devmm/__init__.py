"""devmm — Device Memory Manager.

A uniform, pure-Python interface for allocating and managing device memory
across CPU/CUDA/ROCm, exposing allocations as DLPack >= 1.0 producers.
See specs/devmm-design.md for the full architecture.

Public re-exports (empty, empty_like, Device, Stream, Tensor, DeviceBuffer,
Layout, LayoutPolicy, DeviceMemoryResource, available_runtimes, ...) are
added here as the core modules are implemented.
"""

__version__ = "0.1.0"

__all__: list[str] = []
