---
description: "Critically review the current diff against spec/plan and run the verification gate (make verify), producing a GO / NEEDS-WORK verdict"
---

You are running the review phase for the current feature.

Delegate the work to the **reviewer** subagent
(`.agents/subagents/reviewer.md`). It will:

- Read `spec.md`, `plan.md`, `tasks.md`, and the current diff.
- Run `make verify`.
- Check spec conformance, plan conformance, and implementation
  quality.
- Produce a `GO` or `NEEDS-WORK` verdict with a citation-rich defect
  list (`path/to/file.ext:LINE`) when work remains.

Hand the reviewer's verdict back to the user verbatim. If
`NEEDS-WORK`, the next step is `/build` (Developer role) to address
the defects. If `GO`, summarise what changed since the last verify
(use `git diff --stat` and `git log -1`) and stop.

Never silently skip, disable, or `@ignore` a failing test. If a test
must be skipped, draft an ADR under `docs/adr/` and ask for
confirmation.
