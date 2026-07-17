"""Differential oracle vs NumPy: for every alias with a NumPy counterpart,
itemsize agrees with NumPy's and `np.dtype` objects round-trip through
`DType.from_any` back to the alias (design §3.7, §9).
"""

from __future__ import annotations

import pytest

from devmm import DType

np = pytest.importorskip("numpy")

# bfloat16 is absent: NumPy has no counterpart for it.
NUMPY_ALIAS_NAMES = [
    "bool",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "float16",
    "float32",
    "float64",
    "complex64",
    "complex128",
]


@pytest.mark.parametrize("name", NUMPY_ALIAS_NAMES)
def test_itemsize_matches_numpy(name: str) -> None:
    np_dtype = np.dtype(name)
    assert DType.from_any(np_dtype).itemsize == np_dtype.itemsize


@pytest.mark.parametrize("name", NUMPY_ALIAS_NAMES)
def test_numpy_dtype_round_trips_to_alias(name: str) -> None:
    assert DType.from_any(np.dtype(name)) == DType.from_string(name)
