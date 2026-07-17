#!/usr/bin/env sh
# ensure-toolchain.sh — make `uv` available (idempotent).
# Single source of truth for bootstrapping the build tool; a no-op if
# `uv` is already on PATH.
#
# Consumers:
#   - Claude Code: the SessionStart and Stop hooks in .claude/settings.json.
#   - Claude Code on the web: point the environment setup script here.
#   - Any other agent / CI / devcontainer may call it too.
#
# The install command and URL come from the toolchain_install_* macros in
# _macros.jinja (the single source; docs/tool-bootstrap.md renders the same
# macro for its manual fallback). The installer appends $HOME/.local/bin to the
# shell profile, so the binary is on PATH for *new* shells (e.g. an agent's
# subsequent tool calls), not the current one. A caller that needs it within its
# own already-running shell should also run:
#   export PATH="$HOME/.local/bin:$PATH"
#
# See .agents/README.md for the single-source-of-truth rationale.

command -v uv >/dev/null 2>&1 && exit 0

# A prior install may sit in $HOME/.local/bin without being on this shell's PATH yet.
if [ -x "$HOME/.local/bin/uv" ]; then
	echo "ensure-toolchain: uv is installed at $HOME/.local/bin; add it to PATH (export PATH=\"$HOME/.local/bin:\$PATH\")." >&2
	exit 0
fi

echo "ensure-toolchain: uv not found — installing from https://astral.sh/uv/install.sh ..." >&2
if curl -LsSf https://astral.sh/uv/install.sh | sh >&2 && [ -x "$HOME/.local/bin/uv" ]; then
	echo "ensure-toolchain: installed $("$HOME/.local/bin/uv" --version). On PATH for new shells via your profile." >&2
	exit 0
fi

echo "ensure-toolchain: ERROR — could not install uv automatically (offline or restricted network?)." >&2
echo "ensure-toolchain: install it manually, then retry. See docs/tool-bootstrap.md." >&2
exit 1
