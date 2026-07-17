---
description: Expand spec.md into a numbered, testable plan.md for the current feature
argument-hint: <spec-dir-name> (optional; defaults to the most recent specs/* directory)
---

You are expanding a feature spec into an implementation plan.

1. Identify the target spec directory.
   - If `$ARGUMENTS` is provided, use `specs/$ARGUMENTS/`.
   - Otherwise, use the most recently modified directory under
     `specs/`.
2. Confirm `spec.md` exists and has been reviewed. If it's missing or
   empty, stop and tell the user to run `/spec` first.
3. Delegate the planning to the **architect** subagent
   (`.agents/subagents/architect.md`). It owns the phased-plan format,
   the architecture-decisions block, the "each phase has tests"
   contract, and the "stop and ask before coding" boundary.

The architect subagent will write `plan.md` and mirror it into
`tasks.md`, then stop for user review. Once the user confirms the
plan, the next step is `/build` (Developer role). Do not start
implementing yet.
