"""Public-API snapshot: `devmm`'s exported surface is frozen by this test.

The snapshot maps every name in `devmm.__all__` to its call signature (for
callables and classes) or its type name (for everything else). Growing or
changing the surface requires a matching change in the design doc
(`work/devmm-design.md`) before this snapshot is updated.
"""

from __future__ import annotations

import inspect

import devmm

# The export surface is intentionally empty: nothing is re-exported until the
# corresponding core module lands (design §2).
PUBLIC_API_SNAPSHOT: dict[str, str] = {}


def _describe(obj: object) -> str:
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
