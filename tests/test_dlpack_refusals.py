"""`__dlpack__` refusal paths (design §7.3): `dl_device` mismatch, `copy=True`,
use-after-free, zero-size NULL exports, and the per-platform stream-int
validation table (unit-tested against fakes, no GPU needed).
"""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

from devmm import Device, Stream, Tensor, empty
from devmm._dlpack._abi import DLManagedTensorVersioned
from devmm.mrs.cpu import MallocMemoryResource
from devmm.testing import RecordingMemoryResource

_capsule_get_pointer = ctypes.PYFUNCTYPE(ctypes.c_void_p, ctypes.py_object, ctypes.c_char_p)(
    ("PyCapsule_GetPointer", ctypes.pythonapi)
)


class _FakeGpuStream(Stream):
    """Records synchronize calls; `empty()` and the exporter only ever read
    `device` and ordering primitives off it, so no hardware is involved."""

    def __init__(self, device: Device) -> None:
        self.device = device
        self.synchronize_calls = 0

    @property
    def handle(self) -> int:
        return 0xBEEF

    def synchronize(self) -> None:
        self.synchronize_calls += 1

    def wait_raw(self, other_handle: int) -> None:
        return None


def _cpu_tensor() -> Tensor:
    return empty((2, 3), "float32", mr=MallocMemoryResource())


def _device_tensor(device_str: str) -> tuple[Tensor, _FakeGpuStream]:
    device = Device.from_string(device_str)
    stream = _FakeGpuStream(device)
    t = empty(
        (2, 2),
        "float32",
        device=device,
        mr=RecordingMemoryResource(device=device),
        stream=stream,
    )
    return t, stream


def test_mismatched_dl_device_raises_buffer_error() -> None:
    with pytest.raises(BufferError):
        _cpu_tensor().__dlpack__(dl_device=(2, 0), max_version=(1, 1))


def test_matching_dl_device_is_accepted() -> None:
    t = _cpu_tensor()
    assert t.__dlpack__(dl_device=t.__dlpack_device__(), max_version=(1, 1)) is not None


def test_copy_true_raises_buffer_error() -> None:
    with pytest.raises(BufferError):
        _cpu_tensor().__dlpack__(copy=True, max_version=(1, 1))


@pytest.mark.parametrize("copy", [None, False], ids=["none", "false"])
def test_zero_copy_export_accepts_copy_none_and_false(copy: bool | None) -> None:
    assert _cpu_tensor().__dlpack__(copy=copy, max_version=(1, 1)) is not None


def test_exporting_a_freed_buffer_raises_buffer_error() -> None:
    t = _cpu_tensor()
    t.buffer.free()
    with pytest.raises(BufferError):
        t.__dlpack__(max_version=(1, 1))


def test_zero_size_tensors_export_null_data() -> None:
    t = empty((0, 3), "float32", mr=MallocMemoryResource())
    capsule = t.__dlpack__(max_version=(1, 1))
    ptr = _capsule_get_pointer(capsule, b"dltensor_versioned")
    managed = ctypes.cast(ptr, ctypes.POINTER(DLManagedTensorVersioned)).contents
    assert managed.dl_tensor.data is None


def test_numpy_accepts_zero_size_exports() -> None:
    t = empty((0, 3), "float32", mr=MallocMemoryResource())
    consumed = np.from_dlpack(t)
    assert consumed.shape == (0, 3)
    assert consumed.size == 0


# (device, consumer stream int, accepted?) — design §7.3 per the Array API:
# None means the consumer works on the platform's legacy default stream
# (CUDA: 1, ROCm: 0), which still requires producer ordering; only -1 means
# "do not synchronize". CPU takes None/-1 and nothing else; CUDA disallows 0
# (ambiguous between its two default-stream conventions) and treats 1/2 as
# the legacy/per-thread defaults; ROCm's default is 0 and the CUDA magic
# values 1/2 are meaningless there.
_STREAM_TABLE = [
    ("cpu", None, True),
    ("cpu", -1, True),
    ("cpu", 0, False),
    ("cpu", 1, False),
    ("cpu", 2, False),
    ("cpu", 3, False),
    ("cpu", -2, False),
    ("cuda:0", None, True),
    ("cuda:0", -1, True),
    ("cuda:0", 0, False),
    ("cuda:0", 1, True),
    ("cuda:0", 2, True),
    ("cuda:0", 3, True),
    ("cuda:0", -2, False),
    ("rocm:0", None, True),
    ("rocm:0", -1, True),
    ("rocm:0", 0, True),
    ("rocm:0", 1, False),
    ("rocm:0", 2, False),
    ("rocm:0", 3, True),
    ("rocm:0", -2, False),
]


@pytest.mark.parametrize(("device_str", "stream", "accepted"), _STREAM_TABLE)
def test_stream_int_validation_table(device_str: str, stream: int | None, accepted: bool) -> None:
    if device_str == "cpu":
        t = _cpu_tensor()
    else:
        t, _ = _device_tensor(device_str)
    if accepted:
        assert t.__dlpack__(stream=stream, max_version=(1, 1)) is not None
    else:
        with pytest.raises(BufferError):
            t.__dlpack__(stream=stream, max_version=(1, 1))


def test_non_integer_stream_raises_buffer_error() -> None:
    t, _ = _device_tensor("cuda:0")
    with pytest.raises(BufferError):
        t.__dlpack__(stream="3", max_version=(1, 1))  # type: ignore[arg-type]


def test_consumer_stream_handoff_orders_against_the_producer_stream() -> None:
    t, stream = _device_tensor("cuda:0")
    t.__dlpack__(stream=3, max_version=(1, 1))
    assert stream.synchronize_calls == 1


@pytest.mark.parametrize("device_str", ["cuda:0", "rocm:0"])
def test_none_stream_means_the_legacy_default_and_orders(device_str: str) -> None:
    t, fake = _device_tensor(device_str)
    t.__dlpack__(stream=None, max_version=(1, 1))
    assert fake.synchronize_calls == 1


def test_no_synchronization_when_the_consumer_declines_with_minus_one() -> None:
    t, fake = _device_tensor("cuda:0")
    t.__dlpack__(stream=-1, max_version=(1, 1))
    assert fake.synchronize_calls == 0
