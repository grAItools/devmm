---
description: Create a new feature spec directory under specs/<YYYY-MM>-<slug>/
argument-hint: <kebab-case-slug>
---

You are creating a new feature spec.

1. If `$ARGUMENTS` is empty, ask the user for a slug before creating
   anything.
2. Compute today's date as `YYYY-MM`. The directory is
   `specs/<YYYY-MM>-$ARGUMENTS/`.
3. Create the directory if it doesn't exist. **Do not** pre-create
   `plan.md`, `tasks.md`, or `scratch.md` — each role creates the
   artifacts it owns (Architect: `plan.md`/`tasks.md`; Developer:
   `scratch.md`), and pre-creating them would force those roles to
   `edit` files they should be able to `write` from scratch.
4. Delegate the actual spec authoring to the **product-owner** subagent
   (`.agents/subagents/product-owner.md`). It owns the spec format,
   the testable-criteria rule, the non-goals requirement, and the
   "stop and ask before planning" boundary.

The product-owner subagent will write `spec.md` and then stop for user
review. Once the user confirms the spec, the next step is `/plan`
(Architect role). Do not start implementing yet.
