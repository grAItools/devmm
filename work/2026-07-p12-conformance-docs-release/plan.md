# Plan — Phase 12 — public conformance, docs & release gate

> **Context.** Step 12 of the devmm v0.1 build. Authoritative spec: [`work/devmm-design.md`](../devmm-design.md); build order: [`work/devmm-implementation-plan.md`](../devmm-implementation-plan.md) (Phase 12). Design sections: §9, §10. This step depends on **p11-integrations** landing first.
> The overall v0.1 goals, non-goals and release gate live in the design doc and implementation plan; this folder is scoped to a single phase.

## Architecture decisions
- Promote the two conformance suites to documented public API so external MR/runtime authors get the same guarantees. ADR: n/a (design §9).

## Phase — public conformance, docs & release gate
**Scope.** Promote the conformance entry points, write executed docs + CHANGELOG, build the traceability table, and run the full release checklist.

**Steps.**
1. Public `mr_conformance` + `dlpack_conformance` entry points + tests.
2. README + API docs with every code block executed (doctest); CHANGELOG; bump to `0.1.0`.
3. Build `tests/traceability.md` (every design contract -> >=1 test ID).

**Tests.**
- Release gate 1: `make gate-all` on the full T0 matrix + T1.
- Release gate 2-3: T2 full suite (race canary + torch/cupy); T3 green or signed-off waiver.
- Release gate 4-5: coverage thresholds with reasoned `# pragma: no cover`; refleak + shutdown under `PYTHONDEVMODE=1` + `faulthandler`.
- Release gate 6-8: (recommended) `--with-pydebug` Phase-7 suite; API snapshot matches design + `mypy --strict` + bare-venv; traceability complete.

**Exit criteria.** Release gate green; tag `v0.1.0`.

**Public-API snapshot.** + `testing.mr_conformance`, `testing.dlpack_conformance` (public)

## Risks & open questions
- Unmapped design contract -> traceability table gate.
- Coverage gaming -> reasoned `# pragma: no cover` review.
