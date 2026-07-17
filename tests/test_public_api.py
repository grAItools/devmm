"""Public-API snapshot: `devmm`'s exported surface is frozen by this test.

The snapshot maps every name in `devmm.__all__` to its call signature (for
callables and classes) or its type name (for everything else). Growing or
changing the surface requires a matching change in the design doc
(`work/devmm-design.md`) before this snapshot is updated.
"""

from __future__ import annotations

import enum
import inspect

import devmm

PUBLIC_API_SNAPSHOT: dict[str, str] = {
    "DType": "(code: 'int', bits: 'int', lanes: 'int' = 1) -> None",
    "Device": "(type: 'DeviceType', index: 'int' = 0) -> None",
    "DeviceType": "enum: CPU=1, CUDA=2, CUDA_HOST=3, ROCM=10, ROCM_HOST=11, CUDA_MANAGED=13",
}


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
