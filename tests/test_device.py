"""`Device`/`DeviceType` contract tests: exact DLPack `DLDeviceType` codes,
`from_string` parsing (valid + malformed, hypothesis-fuzzed), format
round-trip, value semantics, and `__dlpack_device__` (design §3.1).
"""

from __future__ import annotations

import dataclasses
import re

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from devmm import Device, DeviceType

# DLDeviceType codes verbatim from dlpack.h.
DLPACK_DEVICE_CODES = {
    "CPU": 1,
    "CUDA": 2,
    "CUDA_HOST": 3,
    "ROCM": 10,
    "ROCM_HOST": 11,
    "CUDA_MANAGED": 13,
}

device_types = st.sampled_from(list(DeviceType))
indices = st.integers(min_value=0, max_value=2**31 - 1)
devices = st.builds(Device, type=device_types, index=indices)

# The full valid-string grammar; anything else must raise ValueError.
_VALID_DEVICE_STRING = re.compile(
    "(" + "|".join(t.name.lower() for t in DeviceType) + ")(:[0-9]+)?"
)


def test_device_type_codes_match_dlpack() -> None:
    assert {member.name: int(member) for member in DeviceType} == DLPACK_DEVICE_CODES


@given(device_type=device_types, index=indices)
def test_from_string_parses_valid(device_type: DeviceType, index: int) -> None:
    parsed = Device.from_string(f"{device_type.name.lower()}:{index}")
    assert parsed == Device(device_type, index)


@given(device_type=device_types)
def test_from_string_bare_name_defaults_to_index_zero(device_type: DeviceType) -> None:
    assert Device.from_string(device_type.name.lower()) == Device(device_type, 0)


@given(device=devices)
def test_from_string_format_round_trip(device: Device) -> None:
    assert Device.from_string(str(device)) == device


@given(text=st.text(max_size=20))
def test_from_string_rejects_malformed(text: str) -> None:
    assume(_VALID_DEVICE_STRING.fullmatch(text) is None)
    with pytest.raises(ValueError):
        Device.from_string(text)


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "cpu:",
        ":0",
        "cuda:-1",
        "cuda:+1",
        "CUDA:0",
        "cuda :0",
        "cuda: 0",
        "cuda:0 ",
        " cpu",
        "gpu:0",
        "cuda:0x1",
        "cuda:1_0",
        "cuda:١",
        "cuda:1\n",
    ],
)
def test_from_string_rejects_malformed_examples(bad: str) -> None:
    with pytest.raises(ValueError):
        Device.from_string(bad)


@given(device=devices)
def test_device_hash_and_equality(device: Device) -> None:
    twin = Device(device.type, device.index)
    assert twin == device
    assert hash(twin) == hash(device)
    assert {device: "hit"}[twin] == "hit"


@given(device=devices)
def test_dlpack_device_is_code_index_pair(device: Device) -> None:
    assert device.__dlpack_device__() == (int(device.type), device.index)


def test_device_index_defaults_to_zero() -> None:
    assert Device(DeviceType.CPU) == Device(DeviceType.CPU, 0)


def test_device_accepts_the_int32_maximum_index() -> None:
    assert Device(DeviceType.CUDA, 2**31 - 1).index == 2**31 - 1


@pytest.mark.parametrize("index", [-1, 2**31, 2**63])
def test_device_rejects_indices_outside_int32(index: int) -> None:
    # DLPack transports the index as an int32_t (`DLDevice.device_id`), so
    # anything wider would be silently truncated at export.
    with pytest.raises(ValueError):
        Device(DeviceType.CUDA, index)


def test_from_string_rejects_an_index_beyond_int32() -> None:
    with pytest.raises(ValueError):
        Device.from_string(f"cuda:{2**31}")


def test_device_is_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        Device(DeviceType.CPU).index = 1  # type: ignore[misc]
