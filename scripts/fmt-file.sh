#!/usr/bin/env bash
# fmt-file.sh — per-file formatter, invoked by the Claude Code
# PostToolUse hook after Write/Edit/MultiEdit with the edited file as $1.
# Keep this fast (<300ms); it runs on every save.
#
# This project uses uv + ruff: apply ruff's autofixes (import sorting and
# other safe fixes) then format, scoped to the single edited file.

set -euo pipefail

file="${1:-}"
[[ -z "$file" ]] && { echo "usage: $0 <file>" >&2; exit 64; }

# Only handle Python sources; silently no-op for anything else so the hook
# stays quiet on non-Python edits (Markdown, config, etc.).
case "$file" in
  *.py | *.pyi) ;;
  *) exit 0 ;;
esac

# The file may have been deleted/renamed between the edit and this hook firing.
[[ -f "$file" ]] || exit 0

# Autofix lint (imports, simple fixes) then format. Quiet, since this runs on
# every save. `--fix` may leave unfixable diagnostics and exit non-zero — that
# is the linter's job to surface via `verify`, not a reason to fail the hook.
uv run ruff check --fix --quiet "$file" || true
uv run ruff format --quiet "$file"
