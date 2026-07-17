# Plan — Phase 4 — CPU memory resources & conformance suite

> **Context.** Step 04 of the devmm v0.1 build. Authoritative spec: [`specs/devmm-design.md`](../devmm-design.md); build order: [`specs/devmm-implementation-plan.md`](../devmm-implementation-plan.md) (Phase 4). Design sections: §5.1. This step depends on **p03-stream-memory-resource** landing first.
> The overall v0.1 goals, non-goals and release gate live in the design doc and implementation plan; this folder is scoped to a single phase.

## Architecture decisions
- MallocMR tracks the allocator family per pointer (POSIX vs Windows) to prevent mismatched frees. ADR: n/a (design §5.1).

## Phase — CPU memory resources & conformance suite
**Scope.** Implement both CPU MRs (incl. the Windows aligned-alloc branch with family tracking) and factor the shared tests into the reusable `devmm.testing.mr_conformance` fixture.

**Steps.**
1. `MallocMemoryResource`: `posix_memalign`/`free` (POSIX), `_aligned_malloc`/`_aligned_free` (Windows), family-tracked; exact `guaranteed_alignment()`.
2. `BytearrayMemoryResource`: over-allocate + offset, pin via buffer export, keep-alive dict; `guaranteed_alignment()==1`.
3. Factor the parametrized fixture -> reusable conformance suite.

**Tests.**
- Write-then-read oracle + no aliasing (both MRs).
- Alignment across hypothesis sizes; `guaranteed_alignment()` honesty.
- Bookkeeping empties (`_debug_live_count()`); double-/unknown-free raise.
- BytearrayMR pinning `BufferError`; weakref dies after free.
- MallocMR leak canary (Linux, `slow`); zero-byte contract.

**Exit criteria.** Gate 4 green; the conformance suite passes for both CPU MRs on all three OSes.

**Public-API snapshot.** + `BytearrayMemoryResource`, `MallocMemoryResource` (via `devmm.mrs.cpu`)

## Risks & open questions
- Windows aligned-free family mismatch -> family-tracking test on the Windows leg.
- Alignment off-by-one -> hypothesis alignment test.
