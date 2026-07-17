"""`DeviceType` (IntEnum mirroring DLPack `DLDeviceType` codes) and the frozen `Device` value
object with `from_string()` and `__dlpack_device__()` (§3.1).
"""

from __future__ import annotations

import dataclasses
import enum
import re


class DeviceType(enum.IntEnum):
    """DLPack `DLDeviceType` codes, verbatim, so no translation table is
    ever needed (design §3.1)."""

    CPU = 1
    CUDA = 2
    CUDA_HOST = 3
    ROCM = 10
    ROCM_HOST = 11
    CUDA_MANAGED = 13


# "name" or "name:index". The index is restricted to ASCII digits so parsing
# accepts exactly what formatting produces (int() would also take "+1",
# "1_0", and non-ASCII digits).
_DEVICE_STRING = re.compile(r"(?P<name>[a-z_]+)(:(?P<index>[0-9]+))?")


@dataclasses.dataclass(frozen=True, slots=True)
class Device:
    """A device identity: DLPack device type plus ordinal index (design §3.1).

    Every buffer, stream and memory resource carries one explicitly — there
    is no ambient "current device" as a correctness mechanism.
    """

    type: DeviceType
    index: int = 0

    @classmethod
    def from_string(cls, s: str) -> Device:
        """Parse "cpu", "cuda:1", "rocm:0", ... into a `Device`.

        Accepted names are the lowercase `DeviceType` member names; a bare
        name means index 0. Anything else raises `ValueError`.
        """
        match = _DEVICE_STRING.fullmatch(s)
        if match is None:
            raise ValueError(
                f"malformed device string {s!r}; expected '<type>' or '<type>:<index>', "
                "e.g. 'cpu' or 'cuda:1'"
            )
        try:
            device_type = DeviceType[match["name"].upper()]
        except KeyError:
            names = ", ".join(member.name.lower() for member in DeviceType)
            raise ValueError(
                f"unknown device type {match['name']!r}; expected one of: {names}"
            ) from None
        index = match["index"]
        return cls(device_type, 0 if index is None else int(index))

    def __str__(self) -> str:
        return f"{self.type.name.lower()}:{self.index}"

    def __dlpack_device__(self) -> tuple[int, int]:
        return (int(self.type), self.index)
