"""NumPy round-trip family — the exporter's primary oracle (design §7, §9).

Hypothesis-generated `(shape, dtype, policy)` triples run against both real
CPU MRs and the runtime-default registry path (`mr=None`, design §4.1);
padded layouts additionally prove byte-correct strides and genuine zero-copy
sharing (mutations travel both ways with no transfer in between).
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from devmm import (
    Aligned,
    ColMajor,
    DeviceOptimal,
    LayoutPolicy,
    Permuted,
    RowMajor,
    empty,
)
from devmm.mrs.cpu import BytearrayMemoryResource, MallocMemoryResource
from tests._dlpack_utils import NUMPY_DTYPE_NAMES, read_back, write_pattern

_MRType = type[BytearrayMemoryResource] | type[MallocMemoryResource] | None

# None exercises `empty()`'s default path: the registry resolves the MR
# through the device runtime (design §3.4/§4.1).
_MR_TYPES: tuple[_MRType, ...] = (BytearrayMemoryResource, MallocMemoryResource, None)
_MR_IDS = ("bytearray", "malloc", "runtime-default")


def _make_mr(mr_type: _MRType) -> BytearrayMemoryResource | MallocMemoryResource | None:
    return None if mr_type is None else mr_type()


@st.composite
def _export_cases(draw: st.DrawFn) -> tuple[tuple[int, ...], str, LayoutPolicy]:
    shape = tuple(draw(st.lists(st.integers(0, 5), max_size=3)))
    policy = draw(
        st.sampled_from(
            [
                RowMajor(),
                ColMajor(),
                DeviceOptimal(),
                Aligned(RowMajor(), unit_stride_alignment=32, base_alignment=64),
                Permuted(tuple(draw(st.permutations(range(len(shape)))))),
            ]
        )
    )
    dtype_name = draw(st.sampled_from(NUMPY_DTYPE_NAMES))
    return shape, dtype_name, policy


@pytest.mark.parametrize("mr_type", _MR_TYPES, ids=_MR_IDS)
@given(case=_export_cases())
def test_numpy_round_trip_matches_dtype_shape_and_values(
    mr_type: _MRType,
    case: tuple[tuple[int, ...], str, LayoutPolicy],
) -> None:
    shape, dtype_name, policy = case
    np_dtype = np.dtype(dtype_name)
    t = empty(shape, dtype_name, layout=policy, mr=_make_mr(mr_type))
    expected = write_pattern(t, np_dtype)
    consumed = np.from_dlpack(t)
    assert consumed.dtype == np_dtype
    assert consumed.shape == shape
    np.testing.assert_array_equal(np.asarray(consumed), expected)


@pytest.mark.parametrize("mr_type", _MR_TYPES, ids=_MR_IDS)
def test_padded_layout_exports_byte_correct_strides(mr_type: _MRType) -> None:
    policy = Aligned(RowMajor(), unit_stride_alignment=64, base_alignment=64)
    t = empty((3, 5), "float32", layout=policy, mr=_make_mr(mr_type))
    # 5 float32s = 20 bytes, padded to the 64-byte line pitch = 16 elements.
    assert t.strides == (16, 1)
    consumed = np.from_dlpack(t)
    assert consumed.strides == (64, 4)


@pytest.mark.parametrize("mr_type", _MR_TYPES, ids=_MR_IDS)
def test_padded_layout_mutations_round_trip_zero_copy(mr_type: _MRType) -> None:
    policy = Aligned(RowMajor(), unit_stride_alignment=64, base_alignment=64)
    np_dtype = np.dtype("float32")
    t = empty((3, 5), "float32", layout=policy, mr=_make_mr(mr_type))
    expected = write_pattern(t, np_dtype)
    consumed = np.from_dlpack(t)

    # Consumer-side mutation is visible through the producer's storage.
    consumed[1, 2] = 99.0
    expected[1, 2] = 99.0
    np.testing.assert_array_equal(read_back(t, np_dtype), expected)

    # Producer-side mutation is visible through the consumer's view.
    rewritten = write_pattern(t, np_dtype)
    np.testing.assert_array_equal(np.asarray(consumed), rewritten)
