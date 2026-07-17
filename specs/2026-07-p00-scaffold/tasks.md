# Tasks — Phase 0 — Scaffold delta & CI skeleton

Mirror of [`plan.md`](plan.md). Tick each box in the same commit as the change.
Write the tests first and observe them fail/skip before implementing. Run the
phase gate before handing off to `/verify`. Do not start the next step until this
one is verified.

## Tests first
- [ ] Test (write first, observe red/skip): `tests/test_public_api.py` — snapshot of `devmm.__all__` (initially empty) + member signatures; green while empty.
- [ ] Test (write first, observe red/skip): `tests/test_packaging.py` — `uv build`, install into a fresh empty venv, `python -c "import devmm"` succeeds (zero-dependency gate step; runs at every gate).
- [ ] Test (write first, observe red/skip): `tests/conftest.py` — `gpu_cuda`/`gpu_rocm` markers + a `recording_mr` fixture placeholder.

## Implement to green
- [ ] Implement: Add `mypy` + `pytest-cov` to the dev group; add `[tool.mypy] strict = true`; keep the `test` extra = `numpy`, `array-api-strict`.
- [ ] Implement: Add `make gate-N` / `make gate-all` targets wrapping lint + `mypy --strict` + tests + wheel-in-bare-venv; keep `make verify` as the hook-invoked cumulative gate.
- [ ] Implement: Add a GitHub Actions workflow: T0 matrix (3 OS x {3.12, latest}) + a T1 Linux job.
- [ ] Implement: Ensure the build backend does not package `tests/`.

## Gate & handoff
- [ ] Update the public-API snapshot: (unchanged; export surface stays empty).
- [ ] Update `tests/traceability.md` for the contracts covered here.
- [ ] Phase gate green (Gate 0 green on the T0 matrix + T1; `__init__.py` re-exports nothing (empty snapshot).).
- [ ] Hand off to `/verify` (Reviewer) before the next step.
