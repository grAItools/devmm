"""Packaging gate: build the wheel, install it into a bare venv, import it.

This is the zero-runtime-dependency enforcement from design §8: `devmm`'s
required dependency set is empty (the CPU MRs and the DLPack layer are
stdlib-only), so the wheel must install and import with nothing else present.
It runs at every gate so dependency creep is caught immediately.
"""

from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

import devmm

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run(args: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(args, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        raise AssertionError(
            f"command failed ({result.returncode}): {args}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


@pytest.fixture(scope="module")
def wheel(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out_dir = tmp_path_factory.mktemp("dist")
    _run(["uv", "build", "--wheel", "--out-dir", str(out_dir)], cwd=PROJECT_ROOT)
    wheels = list(out_dir.glob("*.whl"))
    assert len(wheels) == 1, f"expected exactly one wheel, got {wheels}"
    return wheels[0]


def test_wheel_packages_only_devmm(wheel: Path) -> None:
    with zipfile.ZipFile(wheel) as zf:
        names = zf.namelist()
    packaged_tests = [n for n in names if n.startswith("tests")]
    assert not packaged_tests, f"wheel packages the test suite: {packaged_tests}"
    outside = [n for n in names if not n.startswith(("devmm/", "devmm-"))]
    assert not outside, f"unexpected files in wheel: {outside}"
    assert "devmm/py.typed" in names


def test_wheel_declares_no_runtime_dependencies(wheel: Path) -> None:
    with zipfile.ZipFile(wheel) as zf:
        [metadata_name] = [n for n in zf.namelist() if n.endswith(".dist-info/METADATA")]
        metadata = zf.read(metadata_name).decode()
    requires = [line for line in metadata.splitlines() if line.startswith("Requires-Dist:")]
    # Extras (cuda, rocm, cupy, numba, test) are allowed; unconditional
    # requirements are exactly the dependency creep this gate exists to catch.
    unconditional = [r for r in requires if "extra ==" not in r]
    assert not unconditional, f"wheel declares runtime dependencies: {unconditional}"


def test_wheel_imports_in_bare_venv(wheel: Path, tmp_path: Path) -> None:
    venv_dir = tmp_path / "venv"
    _run(["uv", "venv", "--python", sys.executable, str(venv_dir)])
    python = venv_dir / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    _run(["uv", "pip", "install", "--python", str(python), "--no-deps", str(wheel)])
    result = _run([str(python), "-c", "import devmm; print(devmm.__version__)"])
    assert result.stdout.strip() == devmm.__version__
