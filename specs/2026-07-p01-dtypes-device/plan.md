# Plan — Phase 1 — dtypes & device

> **Context.** Step 01 of the devmm v0.1 build. Authoritative spec: [`specs/devmm-design.md`](../devmm-design.md); build order: [`specs/devmm-implementation-plan.md`](../devmm-implementation-plan.md) (Phase 1). Design sections: §3.1, §3.7. This step depends on **p00-scaffold** landing first.
> The overall v0.1 goals, non-goals and release gate live in the design doc and implementation plan; this folder is scoped to a single phase.

## Architecture decisions
- Duck-type NumPy via `.kind`/`.itemsize`; never import NumPy in module scope. ADR: n/a.

## Phase — dtypes & device
**Scope.** Implement `DType` (frozen `(code, bits, lanes)`, aliases, constructors) and `DeviceType` IntEnum + frozen `Device`.

**Steps.**
1. Implement `DType`: named aliases, itemsize, constructors from Array-API dtype strings and duck-typed NumPy dtypes.
2. Implement `DeviceType` (exact DLPack codes) and frozen `Device` with `from_string`, `__dlpack_device__`.

**Tests.**
- Table-driven alias <-> `(code, bits, lanes)` + itemsize.
- Differential vs NumPy itemsize/round-trip.
- `Device.from_string` hypothesis fuzz (valid + malformed), round-trip, hashability, `__dlpack_device__`.
- Import-hygiene subprocess: no `numpy` in `sys.modules`.

**Exit criteria.** Gate 1 green; snapshot grows to export `DType`, `Device`, `DeviceType`.

**Public-API snapshot.** + `DType`, `Device`, `DeviceType`

## Risks & open questions
- dtype/NumPy divergence -> differential oracle.
- Accidental NumPy import -> subprocess hygiene test.
