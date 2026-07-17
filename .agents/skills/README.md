# Skills

Cross-tool skills live here. Each skill is a directory containing a
`SKILL.md` file (required, with YAML frontmatter `name` and `description`)
and optionally:

- `scripts/` — deterministic executables the agent can invoke instead of
  generating code.
- `references/` — auxiliary docs the agent loads on demand.
- `assets/` — templates, fonts, icons.

The same skills are read by Claude Code (symlinked at `.claude/skills/`)
and OpenCode (symlinked at `.opencode/skills/`).

Write skill descriptions slightly "pushy" — agents tend to under-trigger
skills. Include synonyms ("verify", "check", "is this ready"). The
**Gotchas** section is usually the highest-leverage part.
