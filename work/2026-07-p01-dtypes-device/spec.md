# Phase 1 — dtypes & device

> **Context.** Step 01 of the devmm v0.1 build. Authoritative spec: [`work/devmm-design.md`](../devmm-design.md); build order: [`work/devmm-implementation-plan.md`](../devmm-implementation-plan.md) (Phase 1). Design sections: §3.1, §3.7. This step depends on **p00-scaffold** landing first.
> The overall v0.1 goals, non-goals and release gate live in the design doc and implementation plan; this folder is scoped to a single phase.

## Problem
Nothing yet describes element types or device identity, so no other module can name a dtype or target a device.

## Goal
`DType` and `Device`/`DeviceType` exist, map exactly onto DLPack codes, and construct from Array-API strings and duck-typed NumPy dtypes with no NumPy import at module scope.

## Users & stakeholders
Python developers building on devmm (this phase's consumers are the later phases and the test suite) and the grAItools maintainers who sign off against [`work/devmm-design.md`](../devmm-design.md).

## Success criteria
- Every dtype alias maps to its exact `(code, bits, lanes)` from `dlpack.h` and reports the right itemsize.
- For every alias with a NumPy counterpart, itemsize equals NumPy's and round-trips.
- `Device.from_string` parses well-formed strings, round-trips format, is hashable/equal, and `__dlpack_device__() == (int(type), index)`.
- Importing the dtypes module does not pull `numpy` into `sys.modules`.

## Non-goals
- No layout/stride logic (Phase 2); no runtime/device activation (Phase 8).

## Open questions
- None expected; escalate if any alias lacks a `dlpack.h` code.
