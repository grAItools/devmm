# Phase 2 — layout policies & resolution

> **Context.** Step 02 of the devmm v0.1 build. Authoritative spec: [`specs/devmm-design.md`](../devmm-design.md); build order: [`specs/devmm-implementation-plan.md`](../devmm-implementation-plan.md) (Phase 2). Design sections: §3.6. This step depends on **p01-dtypes-device** landing first.
> The overall v0.1 goals, non-goals and release gate live in the design doc and implementation plan; this folder is scoped to a single phase.

## Problem
There is no way to turn a (shape, dtype) into concrete strides and a byte size, so allocations cannot be sized or exported.

## Goal
`Layout`/`LayoutPolicy` and the shipped policies produce concrete, provably non-overlapping, in-bounds strides for any supported shape.

## Users & stakeholders
Python developers building on devmm (this phase's consumers are the later phases and the test suite) and the grAItools maintainers who sign off against [`specs/devmm-design.md`](../devmm-design.md).

## Success criteria
- For every shipped policy and small shape, all index offsets are distinct, non-negative, and within `required_nbytes` (exhaustive).
- Row-/column-major strides and `nbytes` match NumPy exactly.
- `Aligned` satisfies line-pitch divisibility and minimal padding; `required_nbytes % base_alignment == 0`.
- `layout.base_alignment <= policy.base_alignment` holds and `layout.policy is policy`.
- Policies/layouts are frozen, hashable dict keys; `validate()` rejects overlapping/negative-stride hand-built layouts.

## Non-goals
- No allocation or over-allocation (Phase 4/7); no device-specific alignment lookup beyond `DeviceOptimal`'s declared bounds.

## Open questions
- None expected.
