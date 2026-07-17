"""`DType` contract tests: every alias carries the exact DLPack `DLDataType`
triple, reports the right itemsize, and constructs from Array-API strings and
duck-typed NumPy-like dtype objects (design §3.7).
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest

from devmm import DType
from devmm._core import dtypes

# Expected (code, bits, lanes) triples come verbatim from dlpack.h:
# kDLInt=0, kDLUInt=1, kDLFloat=2, kDLBfloat=4, kDLComplex=5, kDLBool=6.
ALIASES: list[tuple[str, DType, tuple[int, int, int], int]] = [
    ("bool", dtypes.bool_, (6, 8, 1), 1),
    ("int8", dtypes.int8, (0, 8, 1), 1),
    ("int16", dtypes.int16, (0, 16, 1), 2),
    ("int32", dtypes.int32, (0, 32, 1), 4),
    ("int64", dtypes.int64, (0, 64, 1), 8),
    ("uint8", dtypes.uint8, (1, 8, 1), 1),
    ("uint16", dtypes.uint16, (1, 16, 1), 2),
    ("uint32", dtypes.uint32, (1, 32, 1), 4),
    ("uint64", dtypes.uint64, (1, 64, 1), 8),
    ("float16", dtypes.float16, (2, 16, 1), 2),
    ("float32", dtypes.float32, (2, 32, 1), 4),
    ("float64", dtypes.float64, (2, 64, 1), 8),
    ("bfloat16", dtypes.bfloat16, (4, 16, 1), 2),
    ("complex64", dtypes.complex64, (5, 64, 1), 8),
    ("complex128", dtypes.complex128, (5, 128, 1), 16),
]
_IDS = [name for name, _, _, _ in ALIASES]

# NumPy dtype kind characters for every alias with a NumPy counterpart
# (bfloat16 has none). Pairs (kind, itemsize) must select the alias.
NUMPY_KINDS: list[tuple[str, int, DType]] = [
    ("b", 1, dtypes.bool_),
    ("i", 1, dtypes.int8),
    ("i", 2, dtypes.int16),
    ("i", 4, dtypes.int32),
    ("i", 8, dtypes.int64),
    ("u", 1, dtypes.uint8),
    ("u", 2, dtypes.uint16),
    ("u", 4, dtypes.uint32),
    ("u", 8, dtypes.uint64),
    ("f", 2, dtypes.float16),
    ("f", 4, dtypes.float32),
    ("f", 8, dtypes.float64),
    ("c", 8, dtypes.complex64),
    ("c", 16, dtypes.complex128),
]


@pytest.mark.parametrize(("name", "alias", "triple", "itemsize"), ALIASES, ids=_IDS)
def test_alias_matches_dlpack_triple(
    name: str, alias: DType, triple: tuple[int, int, int], itemsize: int
) -> None:
    assert (alias.code, alias.bits, alias.lanes) == triple


@pytest.mark.parametrize(("name", "alias", "triple", "itemsize"), ALIASES, ids=_IDS)
def test_alias_itemsize(
    name: str, alias: DType, triple: tuple[int, int, int], itemsize: int
) -> None:
    assert alias.itemsize == itemsize


@pytest.mark.parametrize(("name", "alias", "triple", "itemsize"), ALIASES, ids=_IDS)
def test_from_string_returns_alias(
    name: str, alias: DType, triple: tuple[int, int, int], itemsize: int
) -> None:
    assert DType.from_string(name) == alias


@pytest.mark.parametrize(("name", "alias", "triple", "itemsize"), ALIASES, ids=_IDS)
def test_from_any_accepts_string_and_dtype(
    name: str, alias: DType, triple: tuple[int, int, int], itemsize: int
) -> None:
    assert DType.from_any(name) == alias
    assert DType.from_any(alias) == alias


@pytest.mark.parametrize(("kind", "itemsize", "alias"), NUMPY_KINDS)
def test_from_any_duck_typed_numpy_like(kind: str, itemsize: int, alias: DType) -> None:
    fake = SimpleNamespace(kind=kind, itemsize=itemsize)
    assert DType.from_any(fake) == alias


@pytest.mark.parametrize("bad", ["", "float128", "int4", "Float32", "bool_", "f4"])
def test_from_string_rejects_unknown(bad: str) -> None:
    with pytest.raises(ValueError):
        DType.from_string(bad)


@pytest.mark.parametrize("kind", ["O", "V", "M", "m", "S", "U"])
def test_from_any_rejects_unsupported_kind(kind: str) -> None:
    with pytest.raises(ValueError):
        DType.from_any(SimpleNamespace(kind=kind, itemsize=8))


@pytest.mark.parametrize("obj", [3.14, 42, object(), None])
def test_from_any_rejects_non_dtype_objects(obj: object) -> None:
    with pytest.raises(TypeError):
        DType.from_any(obj)


def test_dtype_is_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        dtypes.float32.bits = 64  # type: ignore[misc]


def test_dtype_is_hashable_and_equal_by_value() -> None:
    table = {alias: name for name, alias, _, _ in ALIASES}
    assert table[DType(code=2, bits=32, lanes=1)] == "float32"
    assert DType(code=2, bits=32) == DType(code=2, bits=32, lanes=1)
