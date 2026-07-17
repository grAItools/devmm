# Style guide

> One worked example beats a page of prose. Show, don't tell.

## Language

Primary language: **python**. Tooling is configured to enforce
the rules below; this file documents intent, not syntax.

## Conventions

- **Layout & privacy.** Public API is re-exported from `src/devmm/__init__.py`;
  everything under `_core/`, `_dlpack/`, `_runtimes/` is private (leading
  underscore). Users import concrete resources from `devmm.mrs.*` and bridges
  from `devmm.integrations.*`. One class/concept per module, matching the design's
  package layout (§2).
- **Pure Python + `ctypes` only.** No compiled extensions. `ctypes` is confined
  to `_dlpack/` (DLPack structs/capsules) and the runtime FFI shims
  (`_runtimes/*`, the raw-runtime `mrs`); the core domain model imports none of
  it. Never `import numpy` (or any array library) in core — duck-type via
  `.shape` / `.dtype` / `__dlpack_device__`.
- **Value objects are frozen.** `Device`, `Layout`, `DType`, and every
  `LayoutPolicy` are `@dataclass(frozen=True, slots=True)` and hashable.
- **Explicit device, no ambient state.** Every buffer, stream and MR carries an
  explicit `Device`; there is no global "current device" as a *correctness*
  mechanism (context managers are convenience only). See design §3.1.
- **Strong references down the allocator chain.** `DeviceBuffer` → `MR` →
  upstream adaptors are all strong refs, so memory never outlives its allocator.
  GC safety nets use `weakref.finalize` capturing `(mr, ptr, nbytes, stream)` —
  never `self`; never `__del__`.
- **Errors.** `MemoryError` for failed allocation (include device + MR chain in
  the message); `BufferError` for every `__dlpack__` refusal (per spec);
  `RuntimeUnavailableError(ImportError)` for missing runtimes; `StreamError` for
  handoff failures.
- **Typing.** Fully typed, `py.typed` shipped; SPIs are `typing.Protocol`. Start
  every module with `from __future__ import annotations`.
- **Logging.** stdlib `logging` under the `devmm.mr` / `devmm.dlpack` loggers;
  never `print`.

## Anti-patterns (with the fix)

_For every "don't", give the matching "do" right next to it. A wall of "don'ts"
without alternatives makes agents over-cautious and produces worse code._

Example:

> **Don't** catch a bare exception to swallow it.
> **Do** catch the narrowest type that you can recover from, and re-raise
> with `from` so the stack trace stays intact.

## Comments

Comments and docstrings describe the **code**, not the **process** that
produced it. A comment earns its place only when it adds something the code
cannot say for itself.

- **Explain _why_, not _what_.** The code already says what it does. A good
  comment captures what a reader can't recover from the code: an invariant, a
  non-obvious constraint, the spec/ADR that motivates the shape. Don't narrate
  the next line (`increment i`).
- **Keep them true.** A wrong comment is worse than none — it actively
  misleads. When you change behaviour, update or delete the adjacent comment in
  the same change. Tests are the spec; comments get no such safety net, so
  accuracy is on you.
- **No review- or release-process prose.** `out of scope for this PR`,
  `v0.1 covers…`, or internal `Slice X` / `Phase N` labels describe how the
  work was cut up for review. That process moves on and the comment goes stale
  at once. Scope, rationale, and future work belong in the PR description, an
  issue, an ADR, or the spec — _referenced_ from the code, not inlined.
  Reviewers enforce this; if your project wires a comment-content check or
  `ERA`/`FIX`-style lint rules into `make verify`, the gate fails on
  process prose, commented-out code, and stray `TODO`/`FIXME` — file an issue
  instead.
- **State a rationale once.** Don't copy an explanatory block to every call
  site; duplicated prose drifts out of sync. Put it at the definition (or an
  ADR) and reference it.

> **Don't** frame a limitation as review scope:
> `# … out of scope for the minimal v0.1 helper.`
> "v0.1" and "minimal helper" name a milestone that will move; a year on, the
> reader can't tell whether it still holds.
> **Do** state the limitation as a fact about the code, keeping the real
> reason: `# append-mode resume isn't implemented — it needs a pre-sized
> index, so we start at 0.` The behaviour is described; no PR is named. If
> there's tracked future work, link the issue.

## Commit messages

This project uses **[Conventional Commits 1.0.0](https://www.conventionalcommits.org/en/v1.0.0/)**.

Format: `<type>(<scope>): <description>` — the `(scope)` segment
(parentheses included) is optional, so a commit without one is just
`<type>: <description>`. Description in imperative mood, no trailing
period, first line ≤72 chars.

Common types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`,
`build`, `ci`, `perf`, `style`, `revert`. Mark breaking changes with
`!` after the type/scope (e.g. `feat(api)!: drop v1 endpoints`) or a
`BREAKING CHANGE:` footer.

Examples:

    feat(auth): add OIDC token refresh
    fix: handle empty cart in checkout summary
    chore!: drop Node 18 support
    docs(adr): record decision to adopt uv

### Where the convention applies

PRs are **squash-merged**, so only the squash commit lands in history.
Put the Conventional Commits header in the **PR title**. Individual
branch commits during work can be freeform working notes.

## Changelog

When the project keeps a `CHANGELOG.md`, follow
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
[SemVer](https://semver.org/spec/v2.0.0.html):

- Log every user-facing change under `## [Unreleased]`, in the right group
  (`Added` / `Changed` / `Fixed` / `Removed` / `Deprecated` / `Security`).
- Write **one concise bullet** per change, and **lead with the file, command,
  or behaviour** that changed — not the motivation.
- Mark breaking changes (`### Removed (breaking)`, or a `!` per the commit
  convention) and add an `### Upgrade notes` block when an upgrade needs manual
  action.
- Link the ADR when the change has one (see [`docs/adr/`](adr/)).
- On release, rename `[Unreleased]` to the version + date and add the compare
  link.

> **Don't** bury the change under a paragraph of rationale.
> **Do** lead with it: "`find` dropped from the allow-list (its `-delete` form
> bypassed the deny-list)." — the change first, the why in a clause.
