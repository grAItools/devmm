# Tasks — Phase 12 — public conformance, docs & release gate

Mirror of [`plan.md`](plan.md). Tick each box in the same commit as the change.
Write the tests first and observe them fail/skip before implementing. Run the
phase gate before handing off to `/verify`. Do not start the next step until this
one is verified.

## Tests first
- [ ] Test (write first, observe red/skip): Release gate 1: `make gate-all` on the full T0 matrix + T1.
- [ ] Test (write first, observe red/skip): Release gate 2-3: T2 full suite (race canary + torch/cupy); T3 green or signed-off waiver.
- [ ] Test (write first, observe red/skip): Release gate 4-5: coverage thresholds with reasoned `# pragma: no cover`; refleak + shutdown under `PYTHONDEVMODE=1` + `faulthandler`.
- [ ] Test (write first, observe red/skip): Release gate 6-8: (recommended) `--with-pydebug` Phase-7 suite; API snapshot matches design + `mypy --strict` + bare-venv; traceability complete.

## Implement to green
- [ ] Implement: Public `mr_conformance` + `dlpack_conformance` entry points + tests.
- [ ] Implement: README + API docs with every code block executed (doctest); CHANGELOG; bump to `0.1.0`.
- [ ] Implement: Build `tests/traceability.md` (every design contract -> >=1 test ID).

## Gate & handoff
- [ ] Update the public-API snapshot: + `testing.mr_conformance`, `testing.dlpack_conformance` (public).
- [ ] Update `tests/traceability.md` for the contracts covered here.
- [ ] Phase gate green (Release gate green; tag `v0.1.0`.).
- [ ] Hand off to `/verify` (Reviewer) before the next step.
