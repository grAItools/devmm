"""`integrations.rmm.install` over `rmm`-module doubles (design §6, §9),
parametrized across both platform harnesses (hipMM installs under the same
module name, design §4.2): the per-device resource registry round trip
(install -> CallbackMemoryResource, uninstall -> previous resource restored,
exception path included), the callback thunks forwarding to the devmm MR
with translated streams, the platform check, and the direct-cycle guard.
"""

from __future__ import annotations

import sys
from typing import cast

import pytest

from devmm import StatisticsAdaptor
from devmm._runtimes.base import RuntimeUnavailableError
from devmm.integrations import rmm as integrations_rmm
from devmm.testing import RecordingMemoryResource
from tests._gpu_fakes import HARNESSES, FakeRmmMemoryResource
from tests._gpu_fakes import GpuHarness as Harness
from tests._integration_fakes import (
    FakeForeignStream,
    FakeRmmCallbackMemoryResource,
    FakeRmmMr,
    fake_rmm_module,
)


@pytest.fixture(params=sorted(HARNESSES))
def h(request: pytest.FixtureRequest) -> Harness:
    return HARNESSES[cast(str, request.param)]


@pytest.fixture
def fake_rmm(h: Harness, monkeypatch: pytest.MonkeyPatch) -> FakeRmmMr:
    module = fake_rmm_module(h.rmm_marker)
    monkeypatch.setitem(sys.modules, "rmm", module)
    return cast(FakeRmmMr, module.mr)


def _mr(h: Harness) -> RecordingMemoryResource:
    return RecordingMemoryResource(h.device1)


class TestInstallRoundTrip:
    def test_install_replaces_and_uninstall_restores_the_per_device_resource(
        self, h: Harness, fake_rmm: FakeRmmMr
    ) -> None:
        previous = fake_rmm.get_per_device_resource(1)
        handle = integrations_rmm.install(_mr(h))
        installed = fake_rmm.per_device[1]
        assert isinstance(installed, FakeRmmCallbackMemoryResource)
        assert installed is not previous
        handle.uninstall()
        assert fake_rmm.per_device[1] is previous

    def test_context_manager_restores_on_exception(self, h: Harness, fake_rmm: FakeRmmMr) -> None:
        previous = fake_rmm.get_per_device_resource(1)
        with pytest.raises(RuntimeError, match="boom"):  # noqa: SIM117 - the point is nesting
            with integrations_rmm.install(_mr(h)):
                assert fake_rmm.per_device[1] is not previous
                raise RuntimeError("boom")
        assert fake_rmm.per_device[1] is previous

    def test_a_spent_handle_does_not_clobber_a_newer_resource(
        self, h: Harness, fake_rmm: FakeRmmMr
    ) -> None:
        handle = integrations_rmm.install(_mr(h))
        handle.uninstall()
        newer = object()
        fake_rmm.set_per_device_resource(1, newer)
        handle.uninstall()
        assert fake_rmm.per_device[1] is newer

    def test_install_targets_the_mrs_device_index(self, h: Harness, fake_rmm: FakeRmmMr) -> None:
        with integrations_rmm.install(_mr(h)):
            assert 1 in fake_rmm.per_device
            assert 0 not in fake_rmm.per_device


class TestCallbackThunks:
    def test_allocate_forwards_to_the_mr_with_a_translated_stream(
        self, h: Harness, fake_rmm: FakeRmmMr
    ) -> None:
        mr = _mr(h)
        with integrations_rmm.install(mr):
            installed = fake_rmm.per_device[1]
            ptr = installed.allocate_func(64, FakeForeignStream(0xAB))
        name, recorded_ptr, nbytes, stream = mr.calls[0]
        assert (name, nbytes) == ("allocate", 64)
        assert ptr == recorded_ptr
        assert stream.handle == 0xAB
        assert stream.device == h.device1

    def test_deallocate_forwards_to_the_mr(self, h: Harness, fake_rmm: FakeRmmMr) -> None:
        mr = _mr(h)
        with integrations_rmm.install(mr):
            installed = fake_rmm.per_device[1]
            ptr = installed.allocate_func(64, FakeForeignStream(0xAB))
            installed.deallocate_func(ptr, 64, FakeForeignStream(0xCD))
        name, freed_ptr, nbytes, stream = mr.calls[1]
        assert (name, freed_ptr, nbytes) == ("deallocate", ptr, 64)
        assert stream.handle == 0xCD
        assert mr.live == {}

    def test_thunks_accept_a_missing_stream_as_the_default_stream(
        self, h: Harness, fake_rmm: FakeRmmMr
    ) -> None:
        mr = _mr(h)
        with integrations_rmm.install(mr):
            installed = fake_rmm.per_device[1]
            ptr = installed.allocate_func(64)
            installed.deallocate_func(ptr, 64)
        assert [call[3].handle for call in mr.calls] == [0, 0]


class TestRefusals:
    def test_composing_the_consume_and_provide_arrows_raises(
        self, h: Harness, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # No rmm module is importable here: the cycle guard must fire before
        # the import (ValueError, not RuntimeUnavailableError).
        monkeypatch.setitem(sys.modules, "rmm", None)
        mr = h.rmm_mr_cls(FakeRmmMemoryResource(), h.device0)  # type: ignore[call-arg]
        with pytest.raises(ValueError, match="cycle"):
            integrations_rmm.install(mr)

    def test_the_cycle_is_detected_through_the_adaptor_chain(
        self, h: Harness, fake_rmm: FakeRmmMr
    ) -> None:
        inner = h.rmm_mr_cls(FakeRmmMemoryResource(), h.device0)  # type: ignore[call-arg]
        with pytest.raises(ValueError, match="cycle"):
            integrations_rmm.install(StatisticsAdaptor(inner))

    def test_cpu_mrs_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="cuda/rocm"):
            integrations_rmm.install(RecordingMemoryResource())

    def test_a_platform_mismatched_rmm_module_is_refused(
        self, h: Harness, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sibling = HARNESSES["rocm" if h.name == "cuda" else "cuda"]
        monkeypatch.setitem(sys.modules, "rmm", fake_rmm_module(sibling.rmm_marker))
        with pytest.raises(RuntimeUnavailableError, match=h.name):
            integrations_rmm.install(_mr(h))

    def test_a_missing_rmm_module_is_refused(
        self, h: Harness, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A None entry makes `import rmm` raise ImportError, exactly like an
        # absent distribution.
        monkeypatch.setitem(sys.modules, "rmm", None)
        with pytest.raises(RuntimeUnavailableError, match=h.name):
            integrations_rmm.install(_mr(h))
