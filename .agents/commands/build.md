---
description: Implement the current feature one phase at a time, ticking tasks.md and running the verification gate at each phase boundary
argument-hint: <spec-dir-name> (optional; defaults to the most recent specs/* directory)
---

You are carrying out the implementation phase of a feature.

1. Identify the target spec directory.
   - If `$ARGUMENTS` is provided, use `specs/$ARGUMENTS/`.
   - Otherwise, use the most recently modified directory under
     `specs/`.
2. Read `spec.md`, `plan.md`, and `tasks.md` in full. If `plan.md` is
   missing or empty, stop and tell the user to run `/plan` first.
3. If the plan touches an unfamiliar area of the codebase, run an
   `explorer` subagent pass first to summarise the relevant code
   paths, and put the summary in `scratch.md` for the Developer to
   read. (The Developer is itself a subagent and cannot spawn the
   `explorer` subagent on its own.)
4. Delegate the work to the **developer** subagent
   (`.agents/subagents/developer.md`). In the delegation prompt,
   tell it to read `scratch.md` first if you populated it in
   step 3 — that is where the explorer summary lives, and the
   Developer's required-reading rule already covers it when present.
   The developer will then:
   - Work one phase at a time, writing tests first where the plan
     calls for behaviour change.
   - Tick `tasks.md` checkboxes in the same commit as the code change.
   - Run `make verify` at every phase boundary.
   - Stop at the end of each phase and hand off to `/verify`
     (Reviewer) before starting the next.
5. When the developer reports a phase complete, **stop** and ask the
   user to run `/verify` before proceeding. Do not auto-start the
   next phase.

Never silently skip a failing test, edit anything under `*/generated/`,
or run destructive Git. If the plan turns out to be wrong, hand back
to `/plan` (Architect) rather than silently re-planning.
