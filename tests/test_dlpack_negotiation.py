"""Version negotiation (design §7.3): capsule naming, the stamped version,
the read-only flag, and NumPy consuming both the versioned and legacy paths.
"""

from __future__ import annotations

import ctypes
from typing import Any

import numpy as np
import pytest

from devmm import Tensor, empty
from devmm._dlpack._abi import (
    DLPACK_FLAG_BITMASK_READ_ONLY,
    DLManagedTensorVersioned,
)
from devmm.mrs.cpu import MallocMemoryResource
from tests._dlpack_utils import write_pattern

_VERSIONED = b"dltensor_versioned"
_LEGACY = b"dltensor"

# Test-side bindings taking the capsule as an object (the exporter's own use
# them by raw pointer, from inside the capsule destructor).
_capsule_is_valid = ctypes.PYFUNCTYPE(ctypes.c_int, ctypes.py_object, ctypes.c_char_p)(
    ("PyCapsule_IsValid", ctypes.pythonapi)
)
_capsule_get_pointer = ctypes.PYFUNCTYPE(ctypes.c_void_p, ctypes.py_object, ctypes.c_char_p)(
    ("PyCapsule_GetPointer", ctypes.pythonapi)
)


def _tensor(*, read_only: bool = False) -> Tensor:
    t = empty((2, 3), "float32", mr=MallocMemoryResource())
    if read_only:
        t = Tensor(t.buffer, t.dtype, t.shape, t.layout, offset=t.offset, read_only=True)
    return t


def _versioned_struct(capsule: object) -> DLManagedTensorVersioned:
    ptr = _capsule_get_pointer(capsule, _VERSIONED)
    return ctypes.cast(ptr, ctypes.POINTER(DLManagedTensorVersioned)).contents


def test_max_version_1_1_yields_a_versioned_capsule() -> None:
    capsule = _tensor().__dlpack__(max_version=(1, 1))
    assert _capsule_is_valid(capsule, _VERSIONED) == 1
    assert _capsule_is_valid(capsule, _LEGACY) == 0


def test_max_version_none_yields_a_legacy_capsule() -> None:
    capsule = _tensor().__dlpack__()
    assert _capsule_is_valid(capsule, _LEGACY) == 1
    assert _capsule_is_valid(capsule, _VERSIONED) == 0


def test_pre_1_0_max_version_yields_a_legacy_capsule() -> None:
    capsule = _tensor().__dlpack__(max_version=(0, 8))
    assert _capsule_is_valid(capsule, _LEGACY) == 1


@pytest.mark.parametrize(
    ("max_version", "stamped"),
    [((1, 0), (1, 0)), ((1, 1), (1, 1)), ((1, 7), (1, 1)), ((3, 0), (1, 1))],
)
def test_stamped_version_is_min_of_producer_and_consumer(
    max_version: tuple[int, int], stamped: tuple[int, int]
) -> None:
    capsule = _tensor().__dlpack__(max_version=max_version)
    version = _versioned_struct(capsule).version
    assert (version.major, version.minor) == stamped


@pytest.mark.parametrize(
    "max_version",
    [(1, 2, 3), (1,), "11", 7, (1, -3), ("1", "1"), (1.0, 1)],
    ids=["triple", "single", "string", "int", "negative-minor", "digit-strings", "float-major"],
)
def test_malformed_max_version_raises_buffer_error(max_version: object) -> None:
    # A stray string like "11" iterates into two items, and a negative minor
    # would wrap around in the struct's uint32 — both must be refused, not
    # silently stamped.
    with pytest.raises(BufferError):
        _tensor().__dlpack__(max_version=max_version)  # type: ignore[arg-type]


def test_read_only_sets_the_versioned_flag() -> None:
    capsule = _tensor(read_only=True).__dlpack__(max_version=(1, 1))
    assert _versioned_struct(capsule).flags & DLPACK_FLAG_BITMASK_READ_ONLY


def test_writable_exports_carry_no_flags() -> None:
    capsule = _tensor().__dlpack__(max_version=(1, 1))
    assert _versioned_struct(capsule).flags == 0


@pytest.mark.parametrize("max_version", [None, (0, 8)], ids=["none", "pre-1.0"])
def test_read_only_with_a_legacy_consumer_raises_buffer_error(
    max_version: tuple[int, int] | None,
) -> None:
    # The legacy struct cannot express the flag; exporting silently mutable
    # would be unsafe (design §7.3).
    with pytest.raises(BufferError):
        _tensor(read_only=True).__dlpack__(max_version=max_version)


class _ClampedProducer:
    """Forwards NumPy's consumption calls with a pinned `max_version`, driving
    both negotiation paths through `np.from_dlpack` regardless of the version
    NumPy itself asks for."""

    def __init__(self, tensor: Tensor, max_version: tuple[int, int] | None) -> None:
        self._tensor = tensor
        self._max_version = max_version

    def __dlpack_device__(self) -> tuple[int, int]:
        return self._tensor.__dlpack_device__()

    def __dlpack__(
        self,
        *,
        stream: int | None = None,
        max_version: tuple[int, int] | None = None,
        dl_device: tuple[int, int] | None = None,
        copy: bool | None = None,
    ) -> Any:
        return self._tensor.__dlpack__(
            stream=stream, max_version=self._max_version, dl_device=dl_device, copy=copy
        )


@pytest.mark.parametrize("max_version", [None, (1, 1)], ids=["legacy", "versioned"])
def test_numpy_consumes_both_negotiation_paths(max_version: tuple[int, int] | None) -> None:
    t = _tensor()
    expected = write_pattern(t, np.dtype("float32"))
    consumed = np.from_dlpack(_ClampedProducer(t, max_version))
    np.testing.assert_array_equal(np.asarray(consumed), expected)
