"""Import hygiene: the core dtype/device modules duck-type NumPy and must not
pull it into `sys.modules` on import (design §3.7; docs/style.md, "never
import numpy in core"). Checked in a subprocess so the parent test session's
own imports cannot mask a violation.
"""

from __future__ import annotations

import subprocess
import sys


def test_core_import_does_not_import_numpy() -> None:
    code = (
        "import sys\n"
        "import devmm\n"
        "import devmm._core.device\n"
        "import devmm._core.dtypes\n"
        "devmm.DType, devmm.Device, devmm.DeviceType\n"
        "assert 'numpy' not in sys.modules, 'importing devmm core pulled in numpy'\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_integrations_import_pulls_in_no_third_party_library() -> None:
    # The provide arrows mutate third-party state only through explicit
    # install() calls, never as import side effects (design §6) — so
    # importing the whole integrations package must not even import the
    # third-party libraries.
    code = (
        "import sys\n"
        "import devmm.integrations\n"
        "for name in ('numpy', 'cupy', 'numba', 'rmm'):\n"
        "    assert name not in sys.modules, f'importing devmm.integrations pulled in {name}'\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
