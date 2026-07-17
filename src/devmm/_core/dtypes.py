"""`DType`: a frozen 1:1 mapping onto DLPack `DLDataType` (code, bits, lanes) with standard
aliases; duck-types NumPy/Array-API dtypes without importing NumPy (§3.7).
"""

from __future__ import annotations

import dataclasses

# DLDataTypeCode values, verbatim from dlpack.h, so building a DLDataType
# struct never needs a translation table.
_KDL_INT = 0
_KDL_UINT = 1
_KDL_FLOAT = 2
_KDL_BFLOAT = 4
_KDL_COMPLEX = 5
_KDL_BOOL = 6

# NumPy `dtype.kind` characters for the kinds DLPack can express. Absent
# kinds (datetime, strings, object, void, ...) have no DLDataType encoding.
_NUMPY_KIND_TO_CODE = {
    "b": _KDL_BOOL,
    "i": _KDL_INT,
    "u": _KDL_UINT,
    "f": _KDL_FLOAT,
    "c": _KDL_COMPLEX,
}


@dataclasses.dataclass(frozen=True, slots=True)
class DType:
    """An element type as DLPack's `DLDataType` triple (design §3.7).

    Values are stored exactly as `dlpack.h` defines them, so instances drop
    straight into DLPack struct fields. Standard aliases (`float32`,
    `bool_`, ...) live at module level in `devmm._core.dtypes`.
    """

    code: int
    bits: int
    lanes: int = 1

    @property
    def itemsize(self) -> int:
        """Bytes per element across all lanes, rounded up for sub-byte widths."""
        return (self.bits * self.lanes + 7) // 8

    @classmethod
    def from_string(cls, name: str) -> DType:
        """Return the alias named by an Array-API dtype string (e.g. "float32")."""
        try:
            return _ALIASES[name]
        except KeyError:
            raise ValueError(
                f"unknown dtype string {name!r}; expected one of {sorted(_ALIASES)}"
            ) from None

    @classmethod
    def from_any(cls, obj: object) -> DType:
        """Coerce a `DType`, Array-API dtype string, or NumPy-like dtype object.

        NumPy dtypes are duck-typed through `.kind`/`.itemsize` so that no
        array library is ever imported here.
        """
        if isinstance(obj, DType):
            return obj
        if isinstance(obj, str):
            return cls.from_string(obj)
        kind = getattr(obj, "kind", None)
        itemsize = getattr(obj, "itemsize", None)
        if isinstance(kind, str) and isinstance(itemsize, int):
            try:
                code = _NUMPY_KIND_TO_CODE[kind]
            except KeyError:
                raise ValueError(f"dtype kind {kind!r} has no DLPack DLDataType encoding") from None
            return cls(code, itemsize * 8)
        raise TypeError(
            f"cannot interpret {obj!r} as a dtype: expected a DType, an Array-API "
            "dtype string, or an object with `.kind` and `.itemsize`"
        )


bool_ = DType(_KDL_BOOL, 8)
int8 = DType(_KDL_INT, 8)
int16 = DType(_KDL_INT, 16)
int32 = DType(_KDL_INT, 32)
int64 = DType(_KDL_INT, 64)
uint8 = DType(_KDL_UINT, 8)
uint16 = DType(_KDL_UINT, 16)
uint32 = DType(_KDL_UINT, 32)
uint64 = DType(_KDL_UINT, 64)
float16 = DType(_KDL_FLOAT, 16)
float32 = DType(_KDL_FLOAT, 32)
float64 = DType(_KDL_FLOAT, 64)
bfloat16 = DType(_KDL_BFLOAT, 16)
complex64 = DType(_KDL_COMPLEX, 64)
complex128 = DType(_KDL_COMPLEX, 128)

# Array-API canonical spellings only ("bool", not "bool_"): keeping
# `from_string` strict makes accepted inputs predictable and testable.
_ALIASES: dict[str, DType] = {
    "bool": bool_,
    "int8": int8,
    "int16": int16,
    "int32": int32,
    "int64": int64,
    "uint8": uint8,
    "uint16": uint16,
    "uint32": uint32,
    "uint64": uint64,
    "float16": float16,
    "float32": float32,
    "float64": float64,
    "bfloat16": bfloat16,
    "complex64": complex64,
    "complex128": complex128,
}
