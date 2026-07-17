"""Runtime discovery and the CPU runtime (design §4): probe laziness,
entry-point registration, the `DEVMM_RUNTIME` override, the CPU runtime's
SPI surface, and the runtime-backed registry default that completes the CPU
`empty()` story.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Iterator
from importlib import metadata

import numpy as np
import pytest

from devmm import (
    Device,
    DeviceBuffer,
    DeviceType,
    Stream,
    available_runtimes,
    empty,
    get_current_memory_resource,
    runtime_for,
    runtime_names,
)
from devmm._core import registry as registry_module
from devmm._core.stream import CpuStream
from devmm._runtimes import _discovery
from devmm._runtimes.base import CopyKind, DeviceRuntime, RuntimeUnavailableError
from devmm._runtimes.cpu import CpuRuntime
from devmm.mrs.cpu import MallocMemoryResource
from devmm.testing import RecordingMemoryResource
from tests._dlpack_utils import write_pattern

_CPU = Device(DeviceType.CPU)
_CUDA = Device.from_string("cuda:0")

# Structural SPI conformance, verified by mypy --strict: the CPU runtime
# satisfies the `DeviceRuntime` protocol.
_SPI_CHECK: DeviceRuntime = CpuRuntime()


@pytest.fixture(autouse=True)
def _isolated_discovery(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Fresh runtime/spec caches, cpu-only built-ins and no ambient
    `DEVMM_RUNTIME` per test.

    The GPU built-ins probe the host's driver libraries, so leaving them in
    would make these assertions host-dependent; their discovery wiring has
    its own suites (`tests/test_gpu_runtime.py`).
    """
    monkeypatch.delenv("DEVMM_RUNTIME", raising=False)
    monkeypatch.setattr(
        _discovery,
        "_BUILTIN_SPECS",
        tuple(spec for spec in _discovery._BUILTIN_SPECS if spec.name == "cpu"),
    )
    saved = dict(_discovery._loaded)
    _discovery._loaded.clear()
    _discovery._clear_spec_cache()
    yield
    _discovery._loaded.clear()
    _discovery._loaded.update(saved)
    _discovery._clear_spec_cache()


@pytest.fixture
def _isolated_registry() -> Iterator[None]:
    """Snapshot and restore the process-wide current-MR registry."""
    saved = dict(registry_module._registry)
    registry_module._registry.clear()
    yield
    registry_module._registry.clear()
    registry_module._registry.update(saved)


class _FakeGpuStream(Stream):
    """Records synchronize calls; the exporter only reads `device` and the
    ordering primitives, so no hardware is involved."""

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


class FakeCudaRuntime:
    """In-test third-party runtime, registered through the entry-point seam.

    Only the surface discovery and the DLPack handoff touch is implemented:
    identity, `device_types`, and a recording `make_stream_wait`.
    """

    name = "fakecuda"
    device_types = frozenset({DeviceType.CUDA})

    def __init__(self) -> None:
        self.waits: list[tuple[int, Stream]] = []

    def make_stream_wait(self, consumer_handle: int, producer: Stream) -> None:
        self.waits.append((consumer_handle, producer))


def _raising_runtime_factory() -> DeviceRuntime:
    """Loader-convention fake: an installed runtime whose environment check
    fails at load time (see `_discovery._entry_point_probe`)."""
    raise RuntimeUnavailableError("no NVIDIA driver found")


def _install_fake_entry_point(
    monkeypatch: pytest.MonkeyPatch,
    name: str = "fakecuda",
    attr: str = "FakeCudaRuntime",
) -> None:
    entry_point = metadata.EntryPoint(name, f"tests.test_runtimes:{attr}", "devmm.runtimes")
    monkeypatch.setattr(_discovery, "_entry_points", lambda: (entry_point,))
    # The spec table is cached process-wide; the patched seam is only seen
    # after an invalidation.
    _discovery._clear_spec_cache()


class TestDiscovery:
    def test_cpu_runtime_is_always_discovered(self) -> None:
        assert "cpu" in runtime_names()

    def test_runtime_names_probes_without_loading_heavy_modules(self) -> None:
        # Subprocess so the parent session's own imports cannot mask a
        # violation: answering "what's available?" must never pay the
        # runtime-module (or numpy) import cost (design §4.1).
        code = (
            "import sys\n"
            "import devmm\n"
            "names = devmm.runtime_names()\n"
            "assert 'cpu' in names, names\n"
            "heavy = [m for m in ('devmm._runtimes.cpu', 'devmm.mrs.cpu',"
            " 'devmm._runtimes.cuda', 'devmm.mrs.cuda', 'devmm._runtimes.rocm',"
            " 'devmm.mrs.rocm', 'devmm._runtimes._gpulib', 'numpy')"
            " if m in sys.modules]\n"
            "assert not heavy, f'runtime_names() imported {heavy}'\n"
        )
        subprocess.run([sys.executable, "-c", code], check=True)

    def test_available_runtimes_constructs_only_passing_probes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        constructed: list[str] = []

        def load_cpu() -> DeviceRuntime:
            constructed.append("cpu")
            return CpuRuntime()

        def load_absent() -> DeviceRuntime:
            constructed.append("absent")
            raise AssertionError("a failing probe must never be loaded")

        specs = (
            _discovery._RuntimeSpec("cpu", lambda: True, load_cpu),
            _discovery._RuntimeSpec("absent", lambda: False, load_absent),
        )
        monkeypatch.setattr(_discovery, "_BUILTIN_SPECS", specs)
        _discovery._clear_spec_cache()
        assert runtime_names() == ("cpu",)
        assert [runtime.name for runtime in available_runtimes()] == ["cpu"]
        assert constructed == ["cpu"]

    def test_discovery_scans_entry_points_at_most_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Discovery sits on hot paths (host copies, DLPack handoffs), so the
        # entry-point scan must be paid once per process, not per call.
        scans: list[None] = []

        def counting_entry_points() -> tuple[metadata.EntryPoint, ...]:
            scans.append(None)
            return ()

        monkeypatch.setattr(_discovery, "_entry_points", counting_entry_points)
        _discovery._clear_spec_cache()
        runtime_for("cpu")
        runtime_for("cpu")
        runtime_names()
        available_runtimes()
        empty((2, 2), "float32", mr=MallocMemoryResource()).buffer.copy_to_host()
        assert len(scans) == 1

    def test_available_runtimes_loads_the_cpu_runtime(self) -> None:
        assert any(isinstance(runtime, CpuRuntime) for runtime in available_runtimes())

    def test_runtime_for_accepts_device_devicetype_and_string(self) -> None:
        for spec in (_CPU, DeviceType.CPU, "cpu"):
            assert isinstance(runtime_for(spec), CpuRuntime)

    def test_runtime_for_caches_the_loaded_runtime(self) -> None:
        assert runtime_for("cpu") is runtime_for(_CPU)

    def test_runtime_for_unsupported_device_raises_actionably(self) -> None:
        with pytest.raises(RuntimeUnavailableError, match="cuda") as excinfo:
            runtime_for(_CUDA)
        assert "cpu" in str(excinfo.value)


class TestEntryPoints:
    def test_entry_point_runtime_is_discovered_after_builtins(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fake_entry_point(monkeypatch)
        names = runtime_names()
        assert names.index("cpu") < names.index("fakecuda")

    def test_entry_point_runtime_is_loadable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_entry_point(monkeypatch)
        runtime = runtime_for(DeviceType.CUDA)
        assert isinstance(runtime, FakeCudaRuntime)
        assert runtime in available_runtimes()

    def test_builtin_names_shadow_entry_points(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_entry_point(monkeypatch, name="cpu")
        assert runtime_names() == ("cpu",)
        assert isinstance(runtime_for("cpu"), CpuRuntime)

    def test_available_runtimes_skips_runtimes_whose_loader_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fake_entry_point(monkeypatch, name="brokencuda", attr="_raising_runtime_factory")
        # The probe (entry-point presence) passes; only loading reveals the
        # missing environment, and that must not fail the other runtimes.
        assert "brokencuda" in runtime_names()
        assert [runtime.name for runtime in available_runtimes()] == ["cpu"]

    def test_runtime_for_is_not_poisoned_by_an_unrelated_failing_loader(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fake_entry_point(monkeypatch, name="brokencuda", attr="_raising_runtime_factory")
        assert isinstance(runtime_for("cpu"), CpuRuntime)
        # The failure must surface as the actionable no-runtime message for
        # the requested device type, not as the broken loader's own error.
        with pytest.raises(RuntimeUnavailableError, match="supports rocm"):
            runtime_for("rocm:0")

    def test_dlpack_handoff_routes_through_the_runtime(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # With a runtime serving the device, the consumer-stream handoff goes
        # through `runtime.make_stream_wait` instead of the conservative
        # producer synchronize (design §7.3).
        _install_fake_entry_point(monkeypatch)
        runtime = runtime_for(DeviceType.CUDA)
        assert isinstance(runtime, FakeCudaRuntime)
        stream = _FakeGpuStream(_CUDA)
        t = empty(
            (2,),
            "float32",
            device=_CUDA,
            mr=RecordingMemoryResource(device=_CUDA),
            stream=stream,
        )
        t.__dlpack__(stream=3, max_version=(1, 1))
        assert runtime.waits == [(3, stream)]
        assert stream.synchronize_calls == 0


class TestEnvOverride:
    def test_forces_the_named_runtime(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DEVMM_RUNTIME", "cpu")
        assert runtime_names() == ("cpu",)
        assert isinstance(runtime_for("cpu"), CpuRuntime)

    def test_forcing_skips_the_probe(self, monkeypatch: pytest.MonkeyPatch) -> None:
        probed: list[bool] = []

        def probe() -> bool:
            probed.append(True)
            return False

        specs = (_discovery._RuntimeSpec("cpu", probe, _discovery._load_cpu),)
        monkeypatch.setattr(_discovery, "_BUILTIN_SPECS", specs)
        _discovery._clear_spec_cache()
        monkeypatch.setenv("DEVMM_RUNTIME", "cpu")
        assert runtime_names() == ("cpu",)
        assert probed == []

    def test_bogus_value_raises_with_the_registered_names(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DEVMM_RUNTIME", "warpdrive")
        with pytest.raises(RuntimeUnavailableError, match="warpdrive") as excinfo:
            runtime_names()
        message = str(excinfo.value)
        assert "DEVMM_RUNTIME" in message
        assert "cpu" in message

    def test_bogus_value_fails_runtime_for_too(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DEVMM_RUNTIME", "warpdrive")
        with pytest.raises(RuntimeUnavailableError):
            runtime_for("cpu")

    def test_override_excludes_other_runtimes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_entry_point(monkeypatch)
        monkeypatch.setenv("DEVMM_RUNTIME", "cpu")
        with pytest.raises(RuntimeUnavailableError):
            runtime_for(DeviceType.CUDA)


class TestCpuRuntime:
    def test_identity(self) -> None:
        runtime = CpuRuntime()
        assert runtime.name == "cpu"
        assert runtime.device_types == frozenset({DeviceType.CPU})

    def test_device_count(self) -> None:
        runtime = CpuRuntime()
        assert runtime.device_count(DeviceType.CPU) == 1
        assert runtime.device_count(DeviceType.CUDA) == 0

    def test_default_memory_resource_is_malloc(self) -> None:
        mr = CpuRuntime().default_memory_resource(_CPU)
        assert isinstance(mr, MallocMemoryResource)
        assert mr.device == _CPU

    def test_stream_factories_yield_cpu_streams(self) -> None:
        runtime = CpuRuntime()
        for stream in (runtime.default_stream(_CPU), runtime.create_stream(_CPU)):
            assert isinstance(stream, CpuStream)
            assert stream.device == _CPU

    def test_wrap_stream_passes_through_a_cpu_stream(self) -> None:
        stream = CpuStream()
        assert CpuRuntime().wrap_stream(_CPU, stream) is stream

    def test_wrap_stream_wraps_the_null_handle(self) -> None:
        stream = CpuRuntime().wrap_stream(_CPU, 0)
        assert isinstance(stream, CpuStream)
        assert stream.device == _CPU

    def test_wrap_stream_accepts_cuda_stream_protocol_objects(self) -> None:
        class _Proto:
            def __cuda_stream__(self) -> tuple[int, int]:
                return (0, 0)

        assert isinstance(CpuRuntime().wrap_stream(_CPU, _Proto()), CpuStream)

    def test_wrap_stream_rejects_nonzero_handles(self) -> None:
        with pytest.raises(ValueError, match="handle"):
            CpuRuntime().wrap_stream(_CPU, 7)

    def test_wrap_stream_rejects_streams_on_other_devices(self) -> None:
        with pytest.raises(ValueError, match="cuda"):
            CpuRuntime().wrap_stream(_CPU, _FakeGpuStream(_CUDA))

    def test_wrap_stream_rejects_unknown_objects(self) -> None:
        with pytest.raises(TypeError):
            CpuRuntime().wrap_stream(_CPU, object())

    def test_memcpy_moves_host_bytes(self) -> None:
        runtime = CpuRuntime()
        mr = MallocMemoryResource()
        stream = CpuStream()
        pattern = bytes(range(8))
        with (
            DeviceBuffer(8, mr=mr, stream=stream) as src,
            DeviceBuffer(8, mr=mr, stream=stream) as dst,
        ):
            src.copy_from_host(pattern)
            runtime.memcpy(dst.ptr, src.ptr, 8, CopyKind.HOST_TO_HOST, stream)
            assert dst.copy_to_host() == pattern

    def test_memcpy_zero_bytes_is_a_noop(self) -> None:
        runtime = CpuRuntime()
        mr = MallocMemoryResource()
        stream = CpuStream()
        with DeviceBuffer(4, mr=mr, stream=stream) as dst:
            dst.copy_from_host(b"\xaa\xbb\xcc\xdd")
            runtime.memcpy(dst.ptr, 0, 0, CopyKind.HOST_TO_HOST, stream)
            assert dst.copy_to_host() == b"\xaa\xbb\xcc\xdd"

    @pytest.mark.parametrize(
        "kind",
        [
            CopyKind.HOST_TO_DEVICE,
            CopyKind.DEVICE_TO_HOST,
            CopyKind.DEVICE_TO_DEVICE,
            CopyKind.DEFAULT,
        ],
    )
    def test_memcpy_rejects_non_host_kinds(self, kind: CopyKind) -> None:
        with pytest.raises(ValueError, match=kind.name):
            CpuRuntime().memcpy(0, 0, 4, kind, CpuStream())

    def test_memcpy_rejects_negative_sizes(self) -> None:
        with pytest.raises(ValueError, match="-1"):
            CpuRuntime().memcpy(0, 0, -1, CopyKind.HOST_TO_HOST, CpuStream())

    def test_make_stream_wait_is_a_noop(self) -> None:
        # Nothing observable happens; the contract is simply that the call
        # completes for any consumer handle (host work is synchronous).
        CpuRuntime().make_stream_wait(0, CpuStream())
        CpuRuntime().make_stream_wait(0xBEEF, CpuStream())

    def test_activate_device_is_a_noop_context_manager(self) -> None:
        with CpuRuntime().activate_device(_CPU):
            pass

    def test_methods_reject_non_cpu_devices(self) -> None:
        runtime = CpuRuntime()
        with pytest.raises(ValueError, match="cuda"):
            runtime.default_memory_resource(_CUDA)
        with pytest.raises(ValueError, match="cuda"):
            runtime.default_stream(_CUDA)
        with pytest.raises(ValueError, match="cuda"):
            runtime.create_stream(_CUDA)
        with pytest.raises(ValueError, match="cuda"):
            runtime.wrap_stream(_CUDA, 0)
        with pytest.raises(ValueError, match="cuda"):
            runtime.activate_device(_CUDA)

    def test_copy_kind_values_are_the_cuda_hip_codes(self) -> None:
        # cudaMemcpyKind/hipMemcpyKind: H2H, H2D, D2H, D2D, then the UVA
        # direction-inferred Default.
        assert [kind.value for kind in CopyKind] == [0, 1, 2, 3, 4]


@pytest.mark.usefixtures("_isolated_registry")
class TestRuntimeDefaultPath:
    def test_empty_with_no_mr_uses_the_runtime_default(self) -> None:
        t = empty((4, 4), "float32")
        assert isinstance(t.buffer.mr, MallocMemoryResource)
        # The registry caches the runtime default: one MR per device.
        assert t.buffer.mr is get_current_memory_resource(_CPU)
        assert empty((2,), "float32").buffer.mr is t.buffer.mr

    def test_default_path_round_trips_through_numpy(self) -> None:
        np_dtype = np.dtype("float32")
        t = empty((3, 5), "float32")
        expected = write_pattern(t, np_dtype)
        consumed = np.from_dlpack(t)
        assert consumed.dtype == np_dtype
        assert consumed.shape == (3, 5)
        np.testing.assert_array_equal(np.asarray(consumed), expected)
