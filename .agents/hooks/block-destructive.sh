#!/usr/bin/env sh
# block-destructive.sh — canonical destructive-command matcher.
#
# Reads candidate command text on stdin; exits 2 if it matches a forbidden
# pattern, 0 otherwise. This is the single source of truth for the deny-list.
#
# Consumers:
#   - Claude Code: the PreToolUse(Bash) hook in .claude/settings.json pipes the
#     tool input here.
#   - OpenCode: cannot call a script, so the deny globs in
#     .opencode/opencode.jsonc restate these patterns by hand — keep in sync.
#
# See .agents/README.md for the single-source-of-truth rationale.
grep -qE 'rm -rf|push --force|reset --hard|DROP TABLE' && exit 2 || exit 0
