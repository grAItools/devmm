"""Shared pytest configuration: GPU markers and the recording-MR fixture."""

from __future__ import annotations

import os

import pytest

# Marker name -> (DEVMM_GPU opt-in value, description). GPU tests never run by
# default: a silently-passing GPU test on a CPU-only box would be a false green.
_GPU_MARKERS: dict[str, tuple[str, str]] = {
    "gpu_cuda": ("cuda", "requires a CUDA device; opt in with DEVMM_GPU=cuda"),
    "gpu_rocm": ("rocm", "requires a ROCm device; opt in with DEVMM_GPU=rocm"),
}


def pytest_configure(config: pytest.Config) -> None:
    for name, (_, description) in _GPU_MARKERS.items():
        config.addinivalue_line("markers", f"{name}: {description}")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    enabled = os.environ.get("DEVMM_GPU", "")
    for item in items:
        for name, (opt_in, _) in _GPU_MARKERS.items():
            if name in item.keywords and enabled != opt_in:
                item.add_marker(pytest.mark.skip(reason=f"set DEVMM_GPU={opt_in} to run"))


@pytest.fixture
def recording_mr() -> object:
    """The recording memory resource used to assert stream-ordering contracts
    without hardware (design §9).

    `devmm.testing` does not provide `RecordingMemoryResource` yet, so tests
    requesting this fixture are skipped rather than erroring at collection.
    """
    pytest.skip("devmm.testing.RecordingMemoryResource is not implemented")
