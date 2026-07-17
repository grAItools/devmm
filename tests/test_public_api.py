"""Public-API snapshot: `devmm`'s exported surface is frozen by this test.

The snapshots map every name in `devmm.__all__` — and in the public
`devmm.mrs.*` and `devmm.integrations.*` modules users import concrete
memory resources and bridges from — to its call signature (for callables
and classes) or its type name (for everything else). Growing or changing
the surface requires a matching change in the design doc
(`work/devmm-design.md`) before a snapshot is updated.
"""

from __future__ import annotations

import enum
import inspect

import devmm
import devmm.integrations
import devmm.mrs.cpu
import devmm.mrs.cuda
import devmm.mrs.rocm
import devmm.testing

PUBLIC_API_SNAPSHOT: dict[str, str] = {
    "Aligned": (
        "(inner: 'LayoutPolicy', unit_stride_alignment: 'int' = 128, "
        "base_alignment: 'int' = 256) -> None"
    ),
    "CallbackMemoryResource": (
        "(alloc_fn: 'Callable[[int, Stream], int]', "
        "dealloc_fn: 'Callable[[int, int, Stream], None]', device: 'Device') -> 'None'"
    ),
    "ColMajor": "() -> None",
    "DEFAULT": "StreamSentinel",
    "DType": "(code: 'int', bits: 'int', lanes: 'int' = 1) -> None",
    "Device": "(type: 'DeviceType', index: 'int' = 0) -> None",
    "DeviceBuffer": "(nbytes: 'int', *, mr: 'DeviceMemoryResource', stream: 'Stream') -> 'None'",
    "DeviceMemoryResource": "()",
    "DeviceOptimal": "() -> None",
    "DeviceType": "enum: CPU=1, CUDA=2, CUDA_HOST=3, ROCM=10, ROCM_HOST=11, CUDA_MANAGED=13",
    "LEGACY_DEFAULT": "StreamSentinel",
    "Layout": (
        "(permutation: 'tuple[int, ...]', strides: 'tuple[int, ...]', "
        "required_nbytes: 'int', base_alignment: 'int', "
        "policy: 'LayoutPolicy | None' = None) -> None"
    ),
    "LayoutPolicy": "()",
    "LimitingAdaptor": "(upstream: 'DeviceMemoryResource', limit_bytes: 'int') -> 'None'",
    "LoggingAdaptor": (
        "(upstream: 'DeviceMemoryResource', logger: 'logging.Logger | None' = None) -> 'None'"
    ),
    "PER_THREAD_DEFAULT": "StreamSentinel",
    "Permuted": "(permutation: 'tuple[int, ...]') -> None",
    "RowMajor": "() -> None",
    "StatisticsAdaptor": "(upstream: 'DeviceMemoryResource') -> 'None'",
    "Stream": "()",
    "Tensor": (
        "(buffer: 'DeviceBuffer', dtype: 'DType', shape: 'tuple[int, ...]', "
        "layout: 'Layout', *, offset: 'int' = 0, read_only: 'bool' = False) -> 'None'"
    ),
    "available_runtimes": "() -> 'tuple[DeviceRuntime, ...]'",
    "empty": (
        "(shape: 'tuple[int, ...]', dtype: 'object', *, "
        "device: 'Device' = Device(type=<DeviceType.CPU: 1>, index=0), "
        "layout: 'Layout | LayoutPolicy' = DeviceOptimal(), "
        "mr: 'DeviceMemoryResource | None' = None, "
        "stream: 'Stream | None' = None) -> 'Tensor'"
    ),
    "empty_like": (
        "(obj: 'Any', *, dtype: 'object | None' = None, device: 'Device | None' = None, "
        "layout: 'Layout | LayoutPolicy' = DeviceOptimal(), "
        "mr: 'DeviceMemoryResource | None' = None, "
        "stream: 'Stream | None' = None) -> 'Tensor'"
    ),
    "get_current_memory_resource": "(device: 'Device') -> 'DeviceMemoryResource'",
    "runtime_for": "(device: 'Device | DeviceType | str') -> 'DeviceRuntime'",
    "runtime_names": "() -> 'tuple[str, ...]'",
    "set_current_memory_resource": "(mr: 'DeviceMemoryResource') -> 'None'",
    "using_memory_resource": "(mr: 'DeviceMemoryResource') -> 'Iterator[DeviceMemoryResource]'",
}

MRS_CPU_API_SNAPSHOT: dict[str, str] = {
    "BytearrayMemoryResource": (
        "(device: 'Device' = Device(type=<DeviceType.CPU: 1>, index=0), "
        "*, alignment: 'int' = 1) -> 'None'"
    ),
    "MallocMemoryResource": (
        "(device: 'Device' = Device(type=<DeviceType.CPU: 1>, index=0), "
        "*, alignment: 'int' = 64) -> 'None'"
    ),
    "NumpyHandlerMemoryResource": (
        "(device: 'Device' = Device(type=<DeviceType.CPU: 1>, index=0)) -> 'None'"
    ),
}

MRS_CUDA_API_SNAPSHOT: dict[str, str] = {
    "CudaRuntimeMemoryResource": (
        "(device: 'Device', *, async_alloc: \"bool | Literal['auto']\" = 'auto', "
        "api: 'GpuApi | None' = None) -> 'None'"
    ),
    "CupyAllocatorMemoryResource": (
        "(allocator: 'Callable[[int], Any] | None' = None, "
        "device: 'Device' = Device(type=<DeviceType.CUDA: 2>, index=0)) -> 'None'"
    ),
    "RmmMemoryResource": "(inner: 'RmmResourceLike', device: 'Device') -> 'None'",
}

MRS_ROCM_API_SNAPSHOT: dict[str, str] = {
    "HipRuntimeMemoryResource": (
        "(device: 'Device', *, async_alloc: \"bool | Literal['auto']\" = 'auto', "
        "api: 'GpuApi | None' = None) -> 'None'"
    ),
    "HipmmMemoryResource": "(inner: 'RmmResourceLike', device: 'Device') -> 'None'",
}

TESTING_API_SNAPSHOT: dict[str, str] = {
    "RecordingMemoryResource": (
        "(device: 'Device' = Device(type=<DeviceType.CPU: 1>, index=0), "
        "*, stream_ordered: 'bool' = True, guaranteed_alignment: 'int' = 256) -> 'None'"
    ),
    # An exception class over AssertionError's C-level __init__ has no
    # inspectable Python signature.
    "RecordingMisuseError": "<uninspectable callable>",
    "dlpack_conformance": "(device: 'Device') -> 'None'",
    "mr_conformance": (
        "(mr_factory: 'Callable[..., DeviceMemoryResource]', "
        "*, stream_factory: 'Callable[[], Stream] | None' = None, "
        "write: 'Callable[[int, bytes], None] | None' = None, "
        "read: 'Callable[[int, int], bytes] | None' = None) -> 'None'"
    ),
}

INTEGRATIONS_API_SNAPSHOT: dict[str, str] = {
    "Installer": "(library: 'str', restore: 'Callable[[], None]') -> 'None'",
    "cupy": "module",
    "numba": "module",
    "numpy": "module",
    "rmm": "module",
}

_INSTALL_MR_SIGNATURE = "(mr: 'DeviceMemoryResource') -> 'Installer'"

INTEGRATIONS_CUPY_API_SNAPSHOT: dict[str, str] = {"install": _INSTALL_MR_SIGNATURE}
INTEGRATIONS_NUMPY_API_SNAPSHOT: dict[str, str] = {"install": _INSTALL_MR_SIGNATURE}
INTEGRATIONS_RMM_API_SNAPSHOT: dict[str, str] = {"install": _INSTALL_MR_SIGNATURE}

# `DevmmEMMPlugin` is built lazily on first access (it subclasses a Numba
# base class), so this snapshot pins the name list and the eager `install`
# signature; the lazy-class contract itself is pinned in
# tests/test_integrations_numba.py.
INTEGRATIONS_NUMBA_EXPORTS = ["DevmmEMMPlugin", "install"]
INTEGRATIONS_NUMBA_INSTALL_SIGNATURE = "() -> 'Installer'"


def _describe(obj: object) -> str:
    if isinstance(obj, enum.EnumMeta):
        # An enum's surface is its members; for DLPack-code enums the values
        # are ABI. The metaclass call signature (what `inspect.signature`
        # would report) also varies across CPython versions.
        members: list[enum.Enum] = list(obj)
        return "enum: " + ", ".join(f"{member.name}={member.value}" for member in members)
    if callable(obj):
        try:
            return str(inspect.signature(obj))
        except (TypeError, ValueError):
            return "<uninspectable callable>"
    return type(obj).__name__


def test_all_matches_snapshot() -> None:
    assert sorted(devmm.__all__) == sorted(PUBLIC_API_SNAPSHOT)


def test_exported_member_signatures_match_snapshot() -> None:
    described = {name: _describe(getattr(devmm, name)) for name in devmm.__all__}
    assert described == PUBLIC_API_SNAPSHOT


def test_mrs_cpu_all_matches_snapshot() -> None:
    assert sorted(devmm.mrs.cpu.__all__) == sorted(MRS_CPU_API_SNAPSHOT)


def test_mrs_cpu_member_signatures_match_snapshot() -> None:
    described = {name: _describe(getattr(devmm.mrs.cpu, name)) for name in devmm.mrs.cpu.__all__}
    assert described == MRS_CPU_API_SNAPSHOT


def test_mrs_cuda_all_matches_snapshot() -> None:
    assert sorted(devmm.mrs.cuda.__all__) == sorted(MRS_CUDA_API_SNAPSHOT)


def test_mrs_cuda_member_signatures_match_snapshot() -> None:
    described = {name: _describe(getattr(devmm.mrs.cuda, name)) for name in devmm.mrs.cuda.__all__}
    assert described == MRS_CUDA_API_SNAPSHOT


def test_mrs_rocm_all_matches_snapshot() -> None:
    assert sorted(devmm.mrs.rocm.__all__) == sorted(MRS_ROCM_API_SNAPSHOT)


def test_mrs_rocm_member_signatures_match_snapshot() -> None:
    described = {name: _describe(getattr(devmm.mrs.rocm, name)) for name in devmm.mrs.rocm.__all__}
    assert described == MRS_ROCM_API_SNAPSHOT


def test_testing_all_matches_snapshot() -> None:
    assert sorted(devmm.testing.__all__) == sorted(TESTING_API_SNAPSHOT)


def test_testing_member_signatures_match_snapshot() -> None:
    described = {name: _describe(getattr(devmm.testing, name)) for name in devmm.testing.__all__}
    assert described == TESTING_API_SNAPSHOT


def test_integrations_all_matches_snapshot() -> None:
    assert sorted(devmm.integrations.__all__) == sorted(INTEGRATIONS_API_SNAPSHOT)


def test_integrations_member_signatures_match_snapshot() -> None:
    described = {
        name: _describe(getattr(devmm.integrations, name)) for name in devmm.integrations.__all__
    }
    assert described == INTEGRATIONS_API_SNAPSHOT


def test_integrations_cupy_matches_snapshot() -> None:
    assert sorted(devmm.integrations.cupy.__all__) == sorted(INTEGRATIONS_CUPY_API_SNAPSHOT)
    described = {
        name: _describe(getattr(devmm.integrations.cupy, name))
        for name in devmm.integrations.cupy.__all__
    }
    assert described == INTEGRATIONS_CUPY_API_SNAPSHOT


def test_integrations_numpy_matches_snapshot() -> None:
    assert sorted(devmm.integrations.numpy.__all__) == sorted(INTEGRATIONS_NUMPY_API_SNAPSHOT)
    described = {
        name: _describe(getattr(devmm.integrations.numpy, name))
        for name in devmm.integrations.numpy.__all__
    }
    assert described == INTEGRATIONS_NUMPY_API_SNAPSHOT


def test_integrations_rmm_matches_snapshot() -> None:
    assert sorted(devmm.integrations.rmm.__all__) == sorted(INTEGRATIONS_RMM_API_SNAPSHOT)
    described = {
        name: _describe(getattr(devmm.integrations.rmm, name))
        for name in devmm.integrations.rmm.__all__
    }
    assert described == INTEGRATIONS_RMM_API_SNAPSHOT


def test_integrations_numba_matches_snapshot() -> None:
    assert sorted(devmm.integrations.numba.__all__) == sorted(INTEGRATIONS_NUMBA_EXPORTS)
    assert _describe(devmm.integrations.numba.install) == INTEGRATIONS_NUMBA_INSTALL_SIGNATURE
