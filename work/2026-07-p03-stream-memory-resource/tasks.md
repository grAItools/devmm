# Tasks — Phase 3 — stream, memory resource & recording fixture

Mirror of [`plan.md`](plan.md). Tick each box in the same commit as the change.
Write the tests first and observe them fail/skip before implementing. Run the
phase gate before handing off to `/verify`. Do not start the next step until this
one is verified.

## Tests first
- [x] Test (write first, observe red/skip): `RecordingMemoryResource` own tests.
- [x] Test (write first, observe red/skip): Adaptor forwarding + strong-upstream-survives-gc (weakref probe).
- [x] Test (write first, observe red/skip): `StatisticsAdaptor` under hypothesis interleavings + 8-thread x 1k soak.
- [x] Test (write first, observe red/skip): `LimitingAdaptor` boundary-exact; `CallbackMemoryResource` args + exception propagation.
- [x] Test (write first, observe red/skip): `CpuStream` no-ops; sentinel identity.

## Implement to green
- [x] Implement: Build `testing/_recording.py` with its own tests (fake pointers, misuse detection).
- [x] Implement: Implement `Stream` ABC + `CpuStream` + sentinels.
- [x] Implement: Implement `DeviceMemoryResource` ABC (`stream_ordered`, `guaranteed_alignment()`, `available_memory()`).
- [x] Implement: Implement `Statistics`, `Logging`, `Limiting` adaptors and `CallbackMemoryResource`.

## Gate & handoff
- [x] Update the public-API snapshot: + `Stream`, `DeviceMemoryResource`, adaptors, sentinels (per design §2 public surface).
- [x] Update `tests/traceability.md` for the contracts covered here.
- [x] Phase gate green (Gate 3 green; `RecordingMemoryResource` usable as a fixture downstream.).
- [x] Hand off to `/verify` (Reviewer) before the next step.
