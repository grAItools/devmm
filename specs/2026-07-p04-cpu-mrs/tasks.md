# Tasks — Phase 4 — CPU memory resources & conformance suite

Mirror of [`plan.md`](plan.md). Tick each box in the same commit as the change.
Write the tests first and observe them fail/skip before implementing. Run the
phase gate before handing off to `/verify`. Do not start the next step until this
one is verified.

## Tests first
- [ ] Test (write first, observe red/skip): Write-then-read oracle + no aliasing (both MRs).
- [ ] Test (write first, observe red/skip): Alignment across hypothesis sizes; `guaranteed_alignment()` honesty.
- [ ] Test (write first, observe red/skip): Bookkeeping empties (`_debug_live_count()`); double-/unknown-free raise.
- [ ] Test (write first, observe red/skip): BytearrayMR pinning `BufferError`; weakref dies after free.
- [ ] Test (write first, observe red/skip): MallocMR leak canary (Linux, `slow`); zero-byte contract.

## Implement to green
- [ ] Implement: `MallocMemoryResource`: `posix_memalign`/`free` (POSIX), `_aligned_malloc`/`_aligned_free` (Windows), family-tracked; exact `guaranteed_alignment()`.
- [ ] Implement: `BytearrayMemoryResource`: over-allocate + offset, pin via buffer export, keep-alive dict; `guaranteed_alignment()==1`.
- [ ] Implement: Factor the parametrized fixture -> reusable conformance suite.

## Gate & handoff
- [ ] Update the public-API snapshot: + `BytearrayMemoryResource`, `MallocMemoryResource` (via `devmm.mrs.cpu`).
- [ ] Update `tests/traceability.md` for the contracts covered here.
- [ ] Phase gate green (Gate 4 green; the conformance suite passes for both CPU MRs on all three OSes.).
- [ ] Hand off to `/verify` (Reviewer) before the next step.
