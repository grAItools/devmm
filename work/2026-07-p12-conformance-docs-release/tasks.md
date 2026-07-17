# Tasks — Phase 12 — public conformance, docs & release gate

Mirror of [`plan.md`](plan.md). Tick each box in the same commit as the change.
Write the tests first and observe them fail/skip before implementing. Run the
phase gate before handing off to `/verify`. Do not start the next step until this
one is verified.

## Tests first
- [x] Test (write first, observe red/skip): Release gate 1: `make gate-all` on the full T0 matrix + T1.
- [x] Test (write first, observe red/skip): Release gate 2-3: T2 full suite (race canary + torch/cupy); T3 green or signed-off waiver.
- [x] Test (write first, observe red/skip): Release gate 4-5: coverage thresholds with reasoned `# pragma: no cover`; refleak + shutdown under `PYTHONDEVMODE=1` + `faulthandler`.
- [x] Test (write first, observe red/skip): Release gate 6-8: (recommended) `--with-pydebug` Phase-7 suite; API snapshot matches design + `mypy --strict` + bare-venv; traceability complete.

## Implement to green
- [x] Implement: Public `mr_conformance` + `dlpack_conformance` entry points + tests.
- [x] Implement: README + API docs with every code block executed (doctest); CHANGELOG; bump to `0.1.0`.
- [x] Implement: Build `tests/traceability.md` (every design contract -> >=1 test ID).

## Gate & handoff
- [x] Update the public-API snapshot: + `testing.mr_conformance`, `testing.dlpack_conformance` (public).
- [x] Update `tests/traceability.md` for the contracts covered here.
- [x] Phase gate green (Release gate green; tag `v0.1.0`.). T0 legs green locally (`make release-gate`); T2/T3 run under the waiver in `docs/adr/0003-gpu-suite-waiver-for-0.1.0.md` (needs maintainer sign-off); tagging `v0.1.0` is the orchestrator's step.
- [x] Hand off to `/verify` (Reviewer) before the next step.
