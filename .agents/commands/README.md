# Slash commands

Cross-tool slash command definitions live here. Each command is a single
Markdown file with YAML frontmatter (`description`, optional
`argument-hint`).

The same files are read by Claude Code (symlinked at `.claude/commands/`) and
OpenCode (symlinked at `.opencode/commands/`). The symlinks are created by the
post-generation hook; if you copy this layout manually, recreate them with:

```sh
ln -s ../.agents/commands .claude/commands
ln -s ../.agents/commands .opencode/commands
```

Authoring tips: keep each command short and imperative — the description is
what surfaces in the slash-command picker, and the body is the prompt the
agent will follow. Reference `cmd('verify')` and the other shared macros via
`{% from '_macros.jinja' import cmd with context %}` so the wording tracks
the chosen `task_runner`.
