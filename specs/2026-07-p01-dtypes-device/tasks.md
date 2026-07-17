# Tasks — Phase 1 — dtypes & device

Mirror of [`plan.md`](plan.md). Tick each box in the same commit as the change.
Write the tests first and observe them fail/skip before implementing. Run the
phase gate before handing off to `/verify`. Do not start the next step until this
one is verified.

## Tests first
- [ ] Test (write first, observe red/skip): Table-driven alias <-> `(code, bits, lanes)` + itemsize.
- [ ] Test (write first, observe red/skip): Differential vs NumPy itemsize/round-trip.
- [ ] Test (write first, observe red/skip): `Device.from_string` hypothesis fuzz (valid + malformed), round-trip, hashability, `__dlpack_device__`.
- [ ] Test (write first, observe red/skip): Import-hygiene subprocess: no `numpy` in `sys.modules`.

## Implement to green
- [ ] Implement: Implement `DType`: named aliases, itemsize, constructors from Array-API dtype strings and duck-typed NumPy dtypes.
- [ ] Implement: Implement `DeviceType` (exact DLPack codes) and frozen `Device` with `from_string`, `__dlpack_device__`.

## Gate & handoff
- [ ] Update the public-API snapshot: + `DType`, `Device`, `DeviceType`.
- [ ] Update `tests/traceability.md` for the contracts covered here.
- [ ] Phase gate green (Gate 1 green; snapshot grows to export `DType`, `Device`, `DeviceType`.).
- [ ] Hand off to `/verify` (Reviewer) before the next step.
