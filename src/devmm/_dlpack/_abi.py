"""Faithful `ctypes.Structure` mirrors of dlpack.h: DLDevice, DLDataType, DLTensor, DLManagedTensor
(legacy) and DLManagedTensorVersioned, plus flag constants (§7.1).

Field order and types are copied verbatim from the vendored header
(`tests/_abi_oracle/dlpack.h`); the layout is pinned by a compiled-oracle
test and committed per-platform snapshots, so any edit here that shifts a
byte fails the gate before it can corrupt a consumer.

The class-body annotations describe what ctypes' field descriptors return
on *instances* (integers for scalar fields, nested structure views, ...);
the authoritative layout lives in ``_fields_``.
"""

from __future__ import annotations

import ctypes
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # The base class of every CFUNCTYPE (what a function-pointer field
    # returns on instances); typeshed only exposes it from `_ctypes`.
    from _ctypes import CFuncPtr

#: The `(DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION)` of the vendored
#: header these mirrors track.
DLPACK_VERSION: tuple[int, int] = (1, 1)

# DLManagedTensorVersioned.flags bit masks (design §7.1).
DLPACK_FLAG_BITMASK_READ_ONLY = 1 << 0
DLPACK_FLAG_BITMASK_IS_COPIED = 1 << 1


class DLPackVersion(ctypes.Structure):
    """`DLPackVersion`: the ABI version stamped into every versioned capsule."""

    major: int
    minor: int

    _fields_ = [
        ("major", ctypes.c_uint32),
        ("minor", ctypes.c_uint32),
    ]


class DLDevice(ctypes.Structure):
    """`DLDevice`: device type code plus ordinal index.

    `device_type` is a C enum (`DLDeviceType`), which every supported
    64-bit ABI represents as a 32-bit int; `device_id` is `int32_t` in the
    header, which is why `Device` refuses indices outside that range.
    """

    device_type: int
    device_id: int

    _fields_ = [
        ("device_type", ctypes.c_int32),
        ("device_id", ctypes.c_int32),
    ]


class DLDataType(ctypes.Structure):
    """`DLDataType`: `(code, bits, lanes)` — naturally packed into 4 bytes."""

    code: int
    bits: int
    lanes: int

    _fields_ = [
        ("code", ctypes.c_uint8),
        ("bits", ctypes.c_uint8),
        ("lanes", ctypes.c_uint16),
    ]


class DLTensor(ctypes.Structure):
    """`DLTensor`: the plain, non-owning tensor description."""

    data: int | None
    device: DLDevice
    ndim: int
    dtype: DLDataType
    shape: ctypes._Pointer[ctypes.c_int64]
    strides: ctypes._Pointer[ctypes.c_int64]
    byte_offset: int

    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", DLDevice),
        ("ndim", ctypes.c_int32),
        ("dtype", DLDataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]


class DLManagedTensor(ctypes.Structure):
    """Legacy `DLManagedTensor` (pre-1.0 capsules, name ``"dltensor"``)."""

    dl_tensor: DLTensor
    manager_ctx: int | None
    deleter: CFuncPtr


class DLManagedTensorVersioned(ctypes.Structure):
    """`DLManagedTensorVersioned` (DLPack >= 1.0 capsules,
    name ``"dltensor_versioned"``).

    `flags` sits *before* `dl_tensor` so that everything up to and
    including it stays ABI-stable across future versions (per the header's
    note); swapping the two is the classic silent-corruption bug the
    layout tests exist to catch.
    """

    version: DLPackVersion
    manager_ctx: int | None
    deleter: CFuncPtr
    flags: int
    dl_tensor: DLTensor


# The deleters take a pointer to their own struct, so the CFUNCTYPEs (and
# hence ``_fields_``) can only be built after the classes exist.
DLManagedTensorDeleter = ctypes.CFUNCTYPE(None, ctypes.POINTER(DLManagedTensor))
DLManagedTensorVersionedDeleter = ctypes.CFUNCTYPE(None, ctypes.POINTER(DLManagedTensorVersioned))

DLManagedTensor._fields_ = [
    ("dl_tensor", DLTensor),
    ("manager_ctx", ctypes.c_void_p),
    ("deleter", DLManagedTensorDeleter),
]

DLManagedTensorVersioned._fields_ = [
    ("version", DLPackVersion),
    ("manager_ctx", ctypes.c_void_p),
    ("deleter", DLManagedTensorVersionedDeleter),
    ("flags", ctypes.c_uint64),
    ("dl_tensor", DLTensor),
]
