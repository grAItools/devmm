# Plan — Phase 7 — export, Tensor & empty()/empty_like() (critical)

> **Context.** Step 07 of the devmm v0.1 build. Authoritative spec: [`work/devmm-design.md`](../devmm-design.md); build order: [`work/devmm-implementation-plan.md`](../devmm-implementation-plan.md) (Phase 7). Design sections: §3.8, §7. This step depends on **p06-dlpack-abi** landing first.
> The overall v0.1 goals, non-goals and release gate live in the design doc and implementation plan; this folder is scoped to a single phase.

## Architecture decisions
- Single-block capsule `[struct | shape | strides]`; module-global `CFUNCTYPE` deleter with a `Py_IsInitialized` guard; `_Holder` incref/decref via `ctypes.pythonapi` with explicit `argtypes`. ADR: n/a (design §7.2).

## Phase — export, Tensor & empty()/empty_like() (critical)
**Scope.** Capsule builder + deleter + holder chain + capsule destructor + version negotiation + runtime-routed stream-handoff hook (CPU no-op), then `Tensor` and `empty`/`empty_like` with alignment-aware over-allocation.

**Steps.**
1. Exporter, module-scope `CFUNCTYPE` deleter, `_Holder` incref/decref (`argtypes = [py_object]`), capsule destructor honoring the consumed-rename protocol, negotiation.
2. `Tensor` (two protocol methods + introspection); `empty`/`empty_like`.

**Tests.**
- NumPy round-trip family (hypothesis; padded zero-copy mutation).
- Version negotiation (capsule names via `PyCapsule_IsValid`; read-only+legacy -> `BufferError`).
- Deleter lifecycle matrix (a)-(f) on RecordingMR incl. refleak harness, foreign-thread, shutdown.
- Refusals (`dl_device`, `copy=True`, zero-size NULL, stream-int table).
- `empty_like` duck-typing; `array_api_strict.from_dlpack` consumes tensors.

**Exit criteria.** Gate 7 green; refleak + subprocess tests green on all three OSes; snapshot grows `Tensor`, `empty`, `empty_like`.

**Public-API snapshot.** + `Tensor`, `empty`, `empty_like`

## Risks & open questions
- `CFUNCTYPE` deleter GC'd -> module-scope ref + survives-gc test + refleak harness.
- Use-after-free across lifetimes -> the (a)-(f) matrix on RecordingMR.
