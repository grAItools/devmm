# Tasks — Phase 2 — layout policies & resolution

Mirror of [`plan.md`](plan.md). Tick each box in the same commit as the change.
Write the tests first and observe them fail/skip before implementing. Run the
phase gate before handing off to `/verify`. Do not start the next step until this
one is verified.

## Tests first
- [x] Test (write first, observe red/skip): Injectivity oracle (hypothesis, ndim<=5, extents<=8, all policies) — the key correctness test.
- [x] Test (write first, observe red/skip): Differential vs NumPy strides/nbytes for Row/Col-major.
- [x] Test (write first, observe red/skip): `Aligned` postconditions (pitch divisibility, minimal padding).
- [x] Test (write first, observe red/skip): Upper-bound invariant + provenance (`layout.policy is policy`).
- [x] Test (write first, observe red/skip): Frozen/hashable; `validate()` rejects bad layouts; edge cases (ndim=0, zero/one/huge extents, integer-only nbytes).

## Implement to green
- [x] Implement: Resolution helper: permutation -> element strides, innermost padded to the unit-stride alignment; `required_nbytes` rounded to `base_alignment`.
- [x] Implement: Implement the five shipped policies as frozen, hashable callables.
- [x] Implement: Implement frozen `Layout` with `validate()` and `is_contiguous`.

## Gate & handoff
- [x] Update the public-API snapshot: + `Layout`, `LayoutPolicy`, `RowMajor`, `ColMajor`, `Permuted`, `Aligned`, `DeviceOptimal`.
- [x] Update `tests/traceability.md` for the contracts covered here.
- [x] Phase gate green (Gate 2 green.).
- [x] Hand off to `/verify` (Reviewer) before the next step.
