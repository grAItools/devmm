# Subagents

Cross-tool subagent definitions live here. Each subagent is a single Markdown
file with YAML frontmatter. The supported keys are:

- `name` (required) — invocation name; identity comes from this, not the
  filename.
- `description` (required) — used by parent agents to decide when to
  delegate; start with "Use proactively when…" for auto-discovery.
- `model` (optional) — `sonnet` / `opus` / `haiku` / `inherit`.
- `tools` (optional) — Claude Code allowlist. Comma-separated names
  (e.g. `Read, Grep, Glob, Bash`). Claude Code only.
- `permission` (optional) — OpenCode per-action map with keys
  `read` / `write` / `edit` / `bash`, each taking `allow` / `ask` /
  `deny`. `bash` can also be a per-pattern map (e.g.
  `"rg *": allow`, `"*": deny`). OpenCode only — Claude Code ignores
  this field.
- `mode` (optional, **strongly recommended for subagents**) —
  OpenCode-only. One of `primary` / `subagent` / `all`. **Defaults
  to `all`**, which would expose the agent as a top-level primary
  OpenCode agent in addition to a delegated one. Set `mode: subagent`
  to keep it delegation-only. Claude Code ignores this field.

Both `tools` and `permission` are typically declared in the same
frontmatter; each tool reads the field it understands and ignores
the other, so one subagent file works on both surfaces. `mode:
subagent` is similarly OpenCode-only.

The same files are read by Claude Code (symlinked at `.claude/agents/`) and
OpenCode (symlinked at `.opencode/agents/`). The symlinks are created by the
post-generation hook; if you copy this layout manually, recreate them with:

```sh
ln -s ../.agents/subagents .claude/agents
ln -s ../.agents/subagents .opencode/agents
```

A subagent runs in its own context window. Use them to keep heavy
exploration or repetitive review out of the main session's context.

**Note:** Claude Code subagents cannot spawn other subagents. If a
subagent needs to delegate (e.g. to `explorer`), it must hand back to
the main agent via the slash command that invoked it.
