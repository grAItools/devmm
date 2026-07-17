# Plan — Phase 2 — layout policies & resolution

> **Context.** Step 02 of the devmm v0.1 build. Authoritative spec: [`work/devmm-design.md`](../devmm-design.md); build order: [`work/devmm-implementation-plan.md`](../devmm-implementation-plan.md) (Phase 2). Design sections: §3.6. This step depends on **p01-dtypes-device** landing first.
> The overall v0.1 goals, non-goals and release gate live in the design doc and implementation plan; this folder is scoped to a single phase.

## Architecture decisions
- Strides are element-based (DLPack convention); policies are immutable configuration objects. ADR: n/a.

## Phase — layout policies & resolution
**Scope.** `LayoutPolicy` ABC (upper-bound alignment properties), frozen `Layout` (`validate()`, `is_contiguous`), a shared resolution helper, and `RowMajor`/`ColMajor`/`Permuted`/`Aligned`/`DeviceOptimal`.

**Steps.**
1. Resolution helper: permutation -> element strides, innermost padded to the unit-stride alignment; `required_nbytes` rounded to `base_alignment`.
2. Implement the five shipped policies as frozen, hashable callables.
3. Implement frozen `Layout` with `validate()` and `is_contiguous`.

**Tests.**
- Injectivity oracle (hypothesis, ndim<=5, extents<=8, all policies) — the key correctness test.
- Differential vs NumPy strides/nbytes for Row/Col-major.
- `Aligned` postconditions (pitch divisibility, minimal padding).
- Upper-bound invariant + provenance (`layout.policy is policy`).
- Frozen/hashable; `validate()` rejects bad layouts; edge cases (ndim=0, zero/one/huge extents, integer-only nbytes).

**Exit criteria.** Gate 2 green.

**Public-API snapshot.** + `Layout`, `LayoutPolicy`, `RowMajor`, `ColMajor`, `Permuted`, `Aligned`, `DeviceOptimal`

## Risks & open questions
- Stride overlap / OOB -> injectivity oracle.
- Float sneaking into nbytes -> explicit integer assertion.
