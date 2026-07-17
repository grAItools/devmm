# Traceability — contracts to tests

Maps enforced contracts to the tests that pin them. One row per contract;
extend this table as phases land (see `work/devmm-implementation-plan.md`).

| Contract | Source | Test |
| --- | --- | --- |
| Package imports and exposes `__version__` | design §2 | `tests/test_smoke.py::test_package_imports` |
| Public API surface is frozen by snapshot (currently empty) | plan p00, Architecture decisions | `tests/test_public_api.py::test_all_matches_snapshot` |
| Exported member signatures match the snapshot | plan p00, Architecture decisions | `tests/test_public_api.py::test_exported_member_signatures_match_snapshot` |
| Zero runtime dependencies: wheel installs and imports in a bare venv | design §8 | `tests/test_packaging.py::test_wheel_imports_in_bare_venv` |
| Wheel metadata declares no unconditional `Requires-Dist` | design §8 | `tests/test_packaging.py::test_wheel_declares_no_runtime_dependencies` |
| Wheel ships only `devmm/` (incl. `py.typed`), never `tests/` | plan p00, Risks | `tests/test_packaging.py::test_wheel_packages_only_devmm` |
| `gpu_cuda`/`gpu_rocm` tests skip unless `DEVMM_GPU` opts in | design §9 | enforced by `tests/conftest.py` (hook); no pinning test yet |
| `recording_mr` fixture placeholder skips until `devmm.testing` provides it | design §9 | enforced by `tests/conftest.py` (fixture); no pinning test yet |
