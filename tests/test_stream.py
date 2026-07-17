"""Stream contract tests: the `Stream` ABC surface, `CpuStream` no-op
semantics, and identity semantics of the `DEFAULT`/`LEGACY_DEFAULT`/
`PER_THREAD_DEFAULT` sentinels (design §3.2).
"""

from __future__ import annotations

import copy
import inspect
import pickle

import pytest

from devmm import DEFAULT, LEGACY_DEFAULT, PER_THREAD_DEFAULT, Device, Stream
from devmm._core.stream import CpuStream

SENTINELS = (DEFAULT, LEGACY_DEFAULT, PER_THREAD_DEFAULT)


def test_stream_is_abstract() -> None:
    assert inspect.isabstract(Stream)
    assert Stream.__abstractmethods__ == frozenset({"handle", "synchronize", "wait_raw"})


def test_cpu_stream_defaults_to_cpu_device() -> None:
    assert CpuStream().device == Device.from_string("cpu")


def test_cpu_stream_accepts_explicit_cpu_device() -> None:
    device = Device.from_string("cpu:1")
    assert CpuStream(device).device is device


def test_cpu_stream_rejects_non_cpu_devices() -> None:
    with pytest.raises(ValueError, match="cuda:0"):
        CpuStream(Device.from_string("cuda:0"))


def test_cpu_stream_synchronize_and_wait_raw_are_noops() -> None:
    # Nothing to observe on a no-op beyond "returns normally, repeatedly,
    # for any handle" — a raise anywhere here fails the test.
    stream = CpuStream()
    for handle in (0, 0xDEAD):
        stream.synchronize()
        stream.wait_raw(handle)


def test_cpu_stream_handle_is_the_null_handle() -> None:
    assert CpuStream().handle == 0


def test_cpu_stream_implements_the_cuda_stream_protocol() -> None:
    stream = CpuStream()
    assert stream.__cuda_stream__() == (0, stream.handle)


def test_sentinels_are_distinct_singletons() -> None:
    assert len({id(sentinel) for sentinel in SENTINELS}) == len(SENTINELS)


def test_sentinels_are_not_streams() -> None:
    assert not any(isinstance(sentinel, Stream) for sentinel in SENTINELS)


@pytest.mark.parametrize("sentinel", SENTINELS, ids=repr)
def test_sentinel_copy_and_deepcopy_preserve_identity(sentinel: object) -> None:
    assert copy.copy(sentinel) is sentinel
    assert copy.deepcopy(sentinel) is sentinel


@pytest.mark.parametrize("sentinel", SENTINELS, ids=repr)
def test_sentinel_pickle_round_trips_to_the_singleton(sentinel: object) -> None:
    assert pickle.loads(pickle.dumps(sentinel)) is sentinel


def test_sentinels_are_usable_as_dict_keys() -> None:
    table = {DEFAULT: "default", LEGACY_DEFAULT: "legacy", PER_THREAD_DEFAULT: "per-thread"}
    assert table[DEFAULT] == "default"
    assert table[LEGACY_DEFAULT] == "legacy"
    assert table[PER_THREAD_DEFAULT] == "per-thread"


@pytest.mark.parametrize("sentinel", SENTINELS, ids=repr)
def test_sentinel_repr_names_the_module_singleton(sentinel: object) -> None:
    assert repr(sentinel).startswith("devmm.")
