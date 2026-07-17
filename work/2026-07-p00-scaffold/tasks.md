# Tasks — Phase 0 — Scaffold delta & CI skeleton

Mirror of [`plan.md`](plan.md). Tick each box in the same commit as the change.
Write the tests first and observe them fail/skip before implementing. Run the
phase gate before handing off to `/verify`. Do not start the next step until this
one is verified.

## Tests first
- [x] Test (write first, observe red/skip): `tests/test_public_api.py` — snapshot of `devmm.__all__` (initially empty) + member signatures; green while empty.
- [x] Test (write first, observe red/skip): `tests/test_packaging.py` — `uv build`, install into a fresh empty venv, `python -c "import devmm"` succeeds (zero-dependency gate step; runs at every gate).
- [x] Test (write first, observe red/skip): `tests/conftest.py` — `gpu_cuda`/`gpu_rocm` markers + a `recording_mr` fixture placeholder.

## Implement to green
- [x] Implement: Add `mypy` + `pytest-cov` to the dev group; add `[tool.mypy] strict = true`; keep the `test` extra = `numpy`, `array-api-strict`.
- [x] Implement: Add `make gate-N` / `make gate-all` targets wrapping lint + `mypy --strict` + tests + wheel-in-bare-venv; keep `make verify` as the hook-invoked cumulative gate.
- [x] Implement: Add a GitHub Actions workflow: T0 matrix (3 OS x {3.12, latest}) + a T1 Linux job.
- [x] Implement: Ensure the build backend does not package `tests/`.

## Gate & handoff
- [x] Update the public-API snapshot: (unchanged; export surface stays empty).
- [x] Update `tests/traceability.md` for the contracts covered here.
- [x] Phase gate green (Gate 0 green on the T0 matrix + T1; `__init__.py` re-exports nothing (empty snapshot).). Local `make gate-0` green; CI matrix runs on first push (workflow added this phase).
- [x] Hand off to `/verify` (Reviewer) before the next step.
