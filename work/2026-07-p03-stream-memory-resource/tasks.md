# Tasks — Phase 3 — stream, memory resource & recording fixture

Mirror of [`plan.md`](plan.md). Tick each box in the same commit as the change.
Write the tests first and observe them fail/skip before implementing. Run the
phase gate before handing off to `/verify`. Do not start the next step until this
one is verified.

## Tests first
- [ ] Test (write first, observe red/skip): `RecordingMemoryResource` own tests.
- [ ] Test (write first, observe red/skip): Adaptor forwarding + strong-upstream-survives-gc (weakref probe).
- [ ] Test (write first, observe red/skip): `StatisticsAdaptor` under hypothesis interleavings + 8-thread x 1k soak.
- [ ] Test (write first, observe red/skip): `LimitingAdaptor` boundary-exact; `CallbackMemoryResource` args + exception propagation.
- [ ] Test (write first, observe red/skip): `CpuStream` no-ops; sentinel identity.

## Implement to green
- [ ] Implement: Build `testing/_recording.py` with its own tests (fake pointers, misuse detection).
- [ ] Implement: Implement `Stream` ABC + `CpuStream` + sentinels.
- [ ] Implement: Implement `DeviceMemoryResource` ABC (`stream_ordered`, `guaranteed_alignment()`, `available_memory()`).
- [ ] Implement: Implement `Statistics`, `Logging`, `Limiting` adaptors and `CallbackMemoryResource`.

## Gate & handoff
- [ ] Update the public-API snapshot: + `Stream`, `DeviceMemoryResource`, adaptors, sentinels (per design §2 public surface).
- [ ] Update `tests/traceability.md` for the contracts covered here.
- [ ] Phase gate green (Gate 3 green; `RecordingMemoryResource` usable as a fixture downstream.).
- [ ] Hand off to `/verify` (Reviewer) before the next step.
