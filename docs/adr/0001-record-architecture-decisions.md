# 1. Record architecture decisions

## Status

Accepted

## Context

We need a durable record of architecturally significant decisions that both
humans and AI coding agents can read. Plain markdown files in the repository
are the lightest-weight option that satisfies:

- Search by `grep`/`rg` and by file enumeration.
- Diffable in PRs.
- Append-only history (supersession adds a new file).
- Readable by every coding agent we use without special tooling.

## Decision

Use Michael Nygard's ADR format. One file per decision in `docs/adr/`, named
`NNNN-kebab-title.md` (zero-padded to 4 digits). Sections: **Status,
Context, Decision, Consequences**. Supersession is recorded by a new ADR that
references the old one, not by editing history.

Optional: install [`adr-tools`](https://github.com/npryce/adr-tools) (single
shell-script binary) to scaffold new ADRs with `adr new "<title>"`.

## Consequences

- One more file to write per architectural decision. Worth it.
- Decisions become first-class context for both humans and agents.
- The current state of any decision is derivable by reading the latest ADR
  that touches the topic.
