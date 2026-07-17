"""Shared helpers for the DLPack export tests: NumPy dtype mapping and
pattern round-trips through the `DeviceBuffer` host-copy helpers.

The pattern helpers build a NumPy view over host bytes with the tensor's own
strides/offset, so expected values land exactly where the exporter claims the
elements live — any disagreement shows up as a value mismatch in the
round-trip assertions.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from devmm import Tensor

# devmm alias names with a NumPy counterpart (bfloat16 has none).
NUMPY_DTYPE_NAMES = (
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
)


def _strided_view(
    t: Tensor, np_dtype: np.dtype[Any], raw: bytearray | bytes
) -> np.ndarray[Any, Any]:
    return np.ndarray(
        t.shape,
        dtype=np_dtype,
        buffer=raw,
        offset=t.offset * np_dtype.itemsize,
        strides=tuple(s * np_dtype.itemsize for s in t.strides),
    )


def expected_values(t: Tensor, np_dtype: np.dtype[Any]) -> np.ndarray[Any, Any]:
    """The deterministic element pattern `write_pattern` stores in `t`."""
    count = math.prod(t.shape)
    return (np.arange(1, count + 1) % 97).astype(np_dtype).reshape(t.shape)


def write_pattern(t: Tensor, np_dtype: np.dtype[Any]) -> np.ndarray[Any, Any]:
    """Fill `t`'s elements with a deterministic pattern; return the expected array."""
    values = expected_values(t, np_dtype)
    if values.size:
        raw = bytearray(t.buffer.nbytes)
        _strided_view(t, np_dtype, raw)[...] = values
        t.buffer.copy_from_host(bytes(raw))
    return values


def read_back(t: Tensor, np_dtype: np.dtype[Any]) -> np.ndarray[Any, Any]:
    """Read `t`'s elements back out through `copy_to_host`."""
    return _strided_view(t, np_dtype, t.buffer.copy_to_host()).copy()
