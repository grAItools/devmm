# Architecture Decision Records (ADRs)

One file per architecturally significant decision, named
`NNNN-kebab-title.md` (zero-padded to 4 digits, append-only).

## Format (Michael Nygard)

```markdown
# N. <Decision title>

## Status
<Proposed | Accepted | Deprecated | Superseded by ADR M>

## Context
<What is the issue we're seeing that is motivating this decision?>

## Decision
<What we're going to do.>

## Consequences
<What becomes easier, harder, or different as a result?>
```

Supersession adds a new ADR that references the old one; it does not edit
the old file's content. The full historical record is the value.

Optional: install [`adr-tools`](https://github.com/npryce/adr-tools) (single
shell-script binary) and use `adr new "<title>"` to scaffold the next file
with the correct number.
