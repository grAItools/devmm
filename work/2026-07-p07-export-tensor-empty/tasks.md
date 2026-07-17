# Tasks — Phase 7 — export, Tensor & empty()/empty_like() (critical)

Mirror of [`plan.md`](plan.md). Tick each box in the same commit as the change.
Write the tests first and observe them fail/skip before implementing. Run the
phase gate before handing off to `/verify`. Do not start the next step until this
one is verified.

## Tests first
- [x] Test (write first, observe red/skip): NumPy round-trip family (hypothesis; padded zero-copy mutation).
- [x] Test (write first, observe red/skip): Version negotiation (capsule names via `PyCapsule_IsValid`; read-only+legacy -> `BufferError`).
- [x] Test (write first, observe red/skip): Deleter lifecycle matrix (a)-(f) on RecordingMR incl. refleak harness, foreign-thread, shutdown.
- [x] Test (write first, observe red/skip): Refusals (`dl_device`, `copy=True`, zero-size NULL, stream-int table).
- [x] Test (write first, observe red/skip): `empty_like` duck-typing; `array_api_strict.from_dlpack` consumes tensors.

## Implement to green
- [x] Implement: Exporter, module-scope `CFUNCTYPE` deleter, `_Holder` incref/decref (`argtypes = [py_object]`), capsule destructor honoring the consumed-rename protocol, negotiation.
- [x] Implement: `Tensor` (two protocol methods + introspection); `empty`/`empty_like`.

## Gate & handoff
- [x] Update the public-API snapshot: + `Tensor`, `empty`, `empty_like`.
- [x] Update `tests/traceability.md` for the contracts covered here.
- [x] Phase gate green (Gate 7 green; refleak + subprocess tests green on all three OSes; snapshot grows `Tensor`, `empty`, `empty_like`.).
- [x] Hand off to `/verify` (Reviewer) before the next step.
