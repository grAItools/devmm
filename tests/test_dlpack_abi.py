"""DLPack ABI layout tests for the ctypes mirrors in `devmm._dlpack._abi`.

Three independent oracles pin the layout (design §7.1, §9):

- a **compiled** `dlpack.h` (the vendored copy under `tests/_abi_oracle/`)
  whose `sizeof`/`offsetof` output is diffed against ctypes — the
  load-bearing check, run wherever a C compiler exists (T1);
- committed per-platform JSON **snapshots** of that same output, so
  compiler-less machines (T0) still verify the layout;
- a raw **byte-pattern** round-trip through `DLManagedTensorVersioned`
  that catches field reordering even when sizes coincide.
"""

from __future__ import annotations

import ctypes
import json
import os
import platform
import re
import shutil
import struct
import subprocess
from pathlib import Path
from typing import Any

import pytest

from devmm._dlpack import _abi

ORACLE_DIR = Path(__file__).resolve().parent / "_abi_oracle"
SNAPSHOT_DIR = ORACLE_DIR / "snapshots"
VENDORED_HEADER = ORACLE_DIR / "dlpack.h"

# The 64-bit targets the project commits ABI snapshots for.
TARGET_SNAPSHOTS = ("linux-x86_64", "linux-aarch64", "macos-arm64", "windows-x86_64")

MIRRORED_STRUCTS: dict[str, type[ctypes.Structure]] = {
    "DLPackVersion": _abi.DLPackVersion,
    "DLDevice": _abi.DLDevice,
    "DLDataType": _abi.DLDataType,
    "DLTensor": _abi.DLTensor,
    "DLManagedTensor": _abi.DLManagedTensor,
    "DLManagedTensorVersioned": _abi.DLManagedTensorVersioned,
}


def _ctypes_layout() -> dict[str, dict[str, Any]]:
    """The mirrors' layout in the same shape the C oracle emits."""
    layout: dict[str, dict[str, Any]] = {}
    for name, struct_type in MIRRORED_STRUCTS.items():
        fields = {
            field[0]: int(getattr(struct_type, field[0]).offset) for field in struct_type._fields_
        }
        layout[name] = {"size": ctypes.sizeof(struct_type), "fields": fields}
    return layout


def _current_snapshot_name() -> str | None:
    mapping = {
        ("Linux", "x86_64"): "linux-x86_64",
        ("Linux", "aarch64"): "linux-aarch64",
        ("Darwin", "arm64"): "macos-arm64",
        ("Windows", "amd64"): "windows-x86_64",
    }
    return mapping.get((platform.system(), platform.machine().lower()))


def _load_snapshot(name: str) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads((SNAPSHOT_DIR / f"{name}.json").read_text())
    return loaded


def _find_c_compiler() -> str | None:
    for candidate in ("cc", "gcc", "clang"):
        path = shutil.which(candidate)
        if path is not None:
            return path
    return None


def _pinned_version_string() -> str:
    return "{}.{}".format(*_abi.DLPACK_VERSION)


def test_compiled_oracle_matches_ctypes_layout(tmp_path: Path) -> None:
    compiler = _find_c_compiler()
    if compiler is None:
        pytest.skip("no C compiler on PATH (T0); the snapshot test covers layout")
    exe = tmp_path / ("abi_oracle.exe" if os.name == "nt" else "abi_oracle")
    subprocess.run(
        [compiler, "-std=c11", str(ORACLE_DIR / "abi_oracle.c"), "-o", str(exe)],
        check=True,
    )
    emitted = json.loads(
        subprocess.run([str(exe)], check=True, capture_output=True, text=True).stdout
    )
    assert emitted["dlpack_version"] == _pinned_version_string()
    assert emitted["structs"] == _ctypes_layout()


def test_ctypes_layout_matches_the_committed_snapshot() -> None:
    name = _current_snapshot_name()
    if name is None:
        pytest.skip(f"no committed ABI snapshot for {platform.system()}/{platform.machine()}")
    snapshot = _load_snapshot(name)
    assert snapshot["dlpack_version"] == _pinned_version_string()
    assert snapshot["structs"] == _ctypes_layout()


def test_every_target_platform_snapshot_is_committed() -> None:
    for name in TARGET_SNAPSHOTS:
        snapshot = _load_snapshot(name)
        assert snapshot["dlpack_version"] == _pinned_version_string()
        assert snapshot["structs"].keys() == MIRRORED_STRUCTS.keys()


def test_dldatatype_is_packed_with_no_padding() -> None:
    # (uint8, uint8, uint16) must occupy exactly four bytes on every ABI;
    # a stray padding byte would shift `lanes` and corrupt every consumer.
    layout = _ctypes_layout()["DLDataType"]
    assert layout["size"] == 4
    assert layout["fields"] == {"code": 0, "bits": 1, "lanes": 2}


def test_flag_constants_match_the_vendored_header() -> None:
    header = VENDORED_HEADER.read_text()
    defines = {
        match["name"]: 1 << int(match["shift"])
        for match in re.finditer(
            r"#define DLPACK_FLAG_BITMASK_(?P<name>\w+) \(1UL << (?P<shift>\d+)UL\)", header
        )
    }
    assert defines["READ_ONLY"] == _abi.DLPACK_FLAG_BITMASK_READ_ONLY
    assert defines["IS_COPIED"] == _abi.DLPACK_FLAG_BITMASK_IS_COPIED


def test_pinned_version_matches_the_vendored_header() -> None:
    header = VENDORED_HEADER.read_text()
    major = re.search(r"#define DLPACK_MAJOR_VERSION (\d+)", header)
    minor = re.search(r"#define DLPACK_MINOR_VERSION (\d+)", header)
    assert major is not None and minor is not None
    assert (int(major[1]), int(minor[1])) == _abi.DLPACK_VERSION


def test_versioned_field_writes_land_at_the_snapshot_offsets() -> None:
    """Set every `DLManagedTensorVersioned` field by name (nested structs
    included), then decode the raw bytes at the *snapshot* offsets — an
    accidental field reordering in the mirrors moves a write even when the
    total size stays the same, so this fails where a pure size check passes.
    """
    name = _current_snapshot_name()
    if name is None:
        pytest.skip(f"no committed ABI snapshot for {platform.system()}/{platform.machine()}")
    structs = _load_snapshot(name)["structs"]

    manager_ctx = 0x1111_1111_1111_1111
    deleter = 0x2222_2222_2222_2222
    flags = 0x3333_3333_3333_3333
    data = 0x4444_4444_4444_4444
    shape = 0x5555_5555_5555_5555
    strides = 0x6666_6666_6666_6666
    byte_offset = 0x7777_7777_7777_7777

    mtv = _abi.DLManagedTensorVersioned()
    mtv.version.major = 0x0102_0304
    mtv.version.minor = 0x0506_0708
    mtv.manager_ctx = manager_ctx
    mtv.deleter = ctypes.cast(ctypes.c_void_p(deleter), _abi.DLManagedTensorVersionedDeleter)
    mtv.flags = flags
    mtv.dl_tensor.data = data
    mtv.dl_tensor.device.device_type = 0x0A0B_0C0D
    mtv.dl_tensor.device.device_id = 0x1122_3344
    mtv.dl_tensor.ndim = 0x2A2B_2C2D
    mtv.dl_tensor.dtype.code = 0xAB
    mtv.dl_tensor.dtype.bits = 0xCD
    mtv.dl_tensor.dtype.lanes = 0xEF01
    mtv.dl_tensor.shape = ctypes.cast(ctypes.c_void_p(shape), ctypes.POINTER(ctypes.c_int64))
    mtv.dl_tensor.strides = ctypes.cast(ctypes.c_void_p(strides), ctypes.POINTER(ctypes.c_int64))
    mtv.dl_tensor.byte_offset = byte_offset

    raw = bytes(memoryview(mtv))
    assert len(raw) == structs["DLManagedTensorVersioned"]["size"]

    top = structs["DLManagedTensorVersioned"]["fields"]
    version_base = top["version"]
    version_fields = structs["DLPackVersion"]["fields"]
    tensor_base = top["dl_tensor"]
    tensor_fields = structs["DLTensor"]["fields"]
    device_base = tensor_base + tensor_fields["device"]
    device_fields = structs["DLDevice"]["fields"]
    dtype_base = tensor_base + tensor_fields["dtype"]
    dtype_fields = structs["DLDataType"]["fields"]

    # "=" keeps native byte order with fixed-size codes, so the decode is
    # independent of the ctypes field metadata under test.
    expected: list[tuple[str, int, int]] = [
        ("=I", version_base + version_fields["major"], 0x0102_0304),
        ("=I", version_base + version_fields["minor"], 0x0506_0708),
        ("=Q", top["manager_ctx"], manager_ctx),
        ("=Q", top["deleter"], deleter),
        ("=Q", top["flags"], flags),
        ("=Q", tensor_base + tensor_fields["data"], data),
        ("=i", device_base + device_fields["device_type"], 0x0A0B_0C0D),
        ("=i", device_base + device_fields["device_id"], 0x1122_3344),
        ("=i", tensor_base + tensor_fields["ndim"], 0x2A2B_2C2D),
        ("=B", dtype_base + dtype_fields["code"], 0xAB),
        ("=B", dtype_base + dtype_fields["bits"], 0xCD),
        ("=H", dtype_base + dtype_fields["lanes"], 0xEF01),
        ("=Q", tensor_base + tensor_fields["shape"], shape),
        ("=Q", tensor_base + tensor_fields["strides"], strides),
        ("=Q", tensor_base + tensor_fields["byte_offset"], byte_offset),
    ]
    for fmt, offset, value in expected:
        assert struct.unpack_from(fmt, raw, offset)[0] == value, (fmt, offset)
