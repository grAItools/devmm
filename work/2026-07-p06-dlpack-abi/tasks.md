# Tasks — Phase 6 — DLPack ABI mirrors & compiled oracle

Mirror of [`plan.md`](plan.md). Tick each box in the same commit as the change.
Write the tests first and observe them fail/skip before implementing. Run the
phase gate before handing off to `/verify`. Do not start the next step until this
one is verified.

## Tests first
- [x] Test (write first, observe red/skip): T1 compiled oracle (`sizeof`/`offsetof` vs ctypes; skip on T0) — the single most load-bearing test.
- [x] Test (write first, observe red/skip): T0 fallback vs committed JSON snapshots.
- [x] Test (write first, observe red/skip): Field-order regression via `memoryview` byte pattern.

## Implement to green
- [x] Implement: Author the ctypes structs: packed `DLDataType (uint8, uint8, uint16)` with no stray padding; `flags (uint64)` before `dl_tensor` in the versioned struct.
- [x] Implement: Vendor pinned `dlpack.h` + the C oracle emitting JSON; commit `linux-x86_64`, `linux-aarch64`, `macos-arm64`, `windows-x86_64` snapshots.

## Gate & handoff
- [x] Update the public-API snapshot: (no public export; internal `_dlpack/_abi`).
- [x] Update `tests/traceability.md` for the contracts covered here.
- [x] Phase gate green (Gate 6 green; the compiled oracle is green on the T1 CI job.).
- [x] Hand off to `/verify` (Reviewer) before the next step.
