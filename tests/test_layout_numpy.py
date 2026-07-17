"""Differential oracle vs NumPy: `RowMajor`/`ColMajor` element strides and
`required_nbytes` match `np.empty(shape, dtype, order)` exactly for
non-degenerate shapes (design §3.6, §9).
"""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from devmm import ColMajor, Device, DType, RowMajor

np = pytest.importorskip("numpy")

CPU = Device.from_string("cpu")

DTYPE_NAMES = ["uint8", "int16", "float32", "float64", "complex128"]

shapes = st.lists(st.integers(min_value=1, max_value=8), min_size=0, max_size=5).map(tuple)


@given(shape=shapes, name=st.sampled_from(DTYPE_NAMES))
def test_row_major_matches_numpy(shape: tuple[int, ...], name: str) -> None:
    arr = np.empty(shape, dtype=np.dtype(name), order="C")
    layout = RowMajor()(shape, DType.from_string(name), CPU)
    assert layout.strides == tuple(s // arr.itemsize for s in arr.strides)
    assert layout.required_nbytes == arr.nbytes


@given(shape=shapes, name=st.sampled_from(DTYPE_NAMES))
def test_col_major_matches_numpy(shape: tuple[int, ...], name: str) -> None:
    arr = np.empty(shape, dtype=np.dtype(name), order="F")
    layout = ColMajor()(shape, DType.from_string(name), CPU)
    assert layout.strides == tuple(s // arr.itemsize for s in arr.strides)
    assert layout.required_nbytes == arr.nbytes
