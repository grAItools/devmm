# Tasks — Phase 5 — buffer & registry

Mirror of [`plan.md`](plan.md). Tick each box in the same commit as the change.
Write the tests first and observe them fail/skip before implementing. Run the
phase gate before handing off to `/verify`. Do not start the next step until this
one is verified.

## Tests first
- [ ] Test (write first, observe red/skip): Lifecycle vs RecordingMR (deallocate-once, explicit stream, use-after-free).
- [ ] Test (write first, observe red/skip): Finalizer drop/cycle/no-resurrect; interpreter-shutdown subprocess.
- [ ] Test (write first, observe red/skip): Registry strong-ref; restore on exit/exception; contextvar isolation (threads + asyncio); sentinel default raises cleanly.
- [ ] Test (write first, observe red/skip): `copy_from_host`/`copy_to_host` byte-exact round-trip.

## Implement to green
- [ ] Implement: `DeviceBuffer` with the `(mr, ptr, nbytes, stream_handle)` finalizer, `closed` guard, context manager, and `ctypes.memmove` copy helpers.
- [ ] Implement: Registry: strong-ref `dict[Device, MR]`, `using_memory_resource` contextvar override, sentinel lazy default.

## Gate & handoff
- [ ] Update the public-API snapshot: + `DeviceBuffer`, registry accessors (`get`/`set`/`using_memory_resource`).
- [ ] Update `tests/traceability.md` for the contracts covered here.
- [ ] Phase gate green (Gate 5 green.).
- [ ] Hand off to `/verify` (Reviewer) before the next step.
