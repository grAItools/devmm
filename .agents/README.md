# Agent harness

This directory holds the **cross-tool** agent definitions — `subagents/`,
`commands/`, and `skills/` — that are symlinked into each tool's own directory
(`.claude/`, `.opencode/`) so they are defined once and read by all. `hooks/`
holds shared hook scripts (see the deny-list note below).

The governing rule: **root [`AGENTS.md`](../AGENTS.md) is the single source of
truth for agent instructions. Wire every agent to it by reference — never
duplicate content.** Each agent has its own convention for where it looks for
project-level instructions; copying the same guidance into a Claude file, a
Gemini file, a Cursor file, and so on makes the copies drift. Pointing every
agent back at one file prevents that structurally.

## Supported agents

| Agent                                                               | How it reads the instructions                      | Wiring in this repo                                                               |
| ------------------------------------------------------------------- | -------------------------------------------------- | --------------------------------------------------------------------------------- |
| Claude Code                                                         | `CLAUDE.md` (first line `@AGENTS.md`) + `.claude/` | `CLAUDE.md`, `.claude/settings.json`, `.claude/{agents,commands,skills}` symlinks |
| OpenCode                                                            | `.opencode/opencode.jsonc` `instructions`          | `.opencode/opencode.jsonc`, `.opencode/{…}` symlinks                              |
| GitHub Copilot                                                      | native `AGENTS.md` (2026); legacy stub             | `.github/copilot-instructions.md` → `AGENTS.md` (opt-in `copilot`)                |
| OpenAI Codex                                                        | native root `AGENTS.md` (32 KiB doc cap)           | none needed                                                                       |
| Google Gemini CLI                                                   | `.gemini/settings.json` `context.fileName`         | add `.gemini/settings.json` → `AGENTS.md` (see recipe below)                      |
| Jules, Cursor, Windsurf, Roo Code, Zed, JetBrains Junie, Aider, Amp | native root `AGENTS.md`                            | none needed                                                                       |
| Cline, Continue                                                     | a `*-rules/` dir of plain `.md`                    | not wired; see recipe below                                                       |

## Adding an agent

1. **Reads root `AGENTS.md` natively?** Do nothing but add a row above.
2. **Reads a plain-markdown file at a fixed path (no required frontmatter)?**
   Symlink it to `AGENTS.md` with a relative target, like the existing symlinks:
   ```sh
   # examples for rules-directory agents
   ln -s ../AGENTS.md    .clinerules/AGENTS.md
   ln -s ../../AGENTS.md .continue/rules/AGENTS.md
   ```
3. **Needs a config key or a frontmatter'd file?** Add a thin pointer/config stub
   that references `AGENTS.md`. Never copy instruction prose into it. Precedents:
   ```jsonc
   // .github/copilot-instructions.md  → one line: see ../AGENTS.md
   // .gemini/settings.json
   { "context": { "fileName": ["AGENTS.md", "GEMINI.md"] } }
   ```
4. Put tool-specific guidance (not meant for every agent) in that tool's own file,
   not in `AGENTS.md`.

## Caveats

- **Copier-managed.** This harness is generated from a Copier template
  (`gh:grAItools/harness-copier-template`; see `.copier-answers.yml`). Edits to
  template-owned files (`.claude/settings.json`, `.opencode/opencode.jsonc`,
  `AGENTS.md`, the managed `.gitignore` block) can be reverted by `copier update`;
  port durable changes upstream behind a per-agent toggle. Net-new files (this
  README, `.agents/hooks/*`, `.gemini/settings.json`) are safe.
- **Symlinks need `core.symlinks=true`.** On Windows checkouts without it, Git
  materializes a symlink as a text file containing the target path; prefer a stub
  there. The repo already relies on symlinks for `.claude/` / `.opencode/`.
- **Destructive-command deny-list** is canonical in
  [`hooks/block-destructive.sh`](hooks/block-destructive.sh); OpenCode's deny globs
  are a hand-kept mirror (it cannot call a script).
