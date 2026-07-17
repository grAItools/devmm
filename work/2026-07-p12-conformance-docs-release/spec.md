# Phase 12 — public conformance, docs & release gate

> **Context.** Step 12 of the devmm v0.1 build. Authoritative spec: [`work/devmm-design.md`](../devmm-design.md); build order: [`work/devmm-implementation-plan.md`](../devmm-implementation-plan.md) (Phase 12). Design sections: §9, §10. This step depends on **p11-integrations** landing first.
> The overall v0.1 goals, non-goals and release gate live in the design doc and implementation plan; this folder is scoped to a single phase.

## Problem
Third-party MR/runtime authors have no supported way to prove conformance, the library is undocumented, and there is no single release checklist mapping the design to tests.

## Goal
The conformance suites are public, docs are executed as tests, and a release gate maps every design contract to a test before tagging 0.1.0.

## Users & stakeholders
Python developers building on devmm (this phase's consumers are the later phases and the test suite) and the grAItools maintainers who sign off against [`work/devmm-design.md`](../devmm-design.md).

## Success criteria
- `devmm.testing.mr_conformance(mr_factory)` and `devmm.testing.dlpack_conformance(device)` are public, documented, and self-tested.
- Every code block in README + API docs executes green under doctest.
- `tests/traceability.md` maps every design "must/raises/contract" sentence to at least one test ID, with no unmapped sentence remaining.
- The full release gate is green: gate-all on T0 matrix + T1; T2 suite (race canary + torch/cupy); T3 green or a signed-off waiver; coverage >=95% core/dlpack and >=90% overall; refleak/shutdown under `PYTHONDEVMODE=1`; API snapshot matches the design; `mypy --strict` and bare-venv green.

## Non-goals
- No new library features; the v2 candidates in design §10 stay deferred.

## Open questions
- Final decision on the T3/ROCm waiver; whether the `--with-pydebug` job is required or recommended.
