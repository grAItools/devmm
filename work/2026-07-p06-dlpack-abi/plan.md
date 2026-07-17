# Plan — Phase 6 — DLPack ABI mirrors & compiled oracle

> **Context.** Step 06 of the devmm v0.1 build. Authoritative spec: [`work/devmm-design.md`](../devmm-design.md); build order: [`work/devmm-implementation-plan.md`](../devmm-implementation-plan.md) (Phase 6). Design sections: §7.1. This step depends on **p05-buffer-registry** landing first.
> The overall v0.1 goals, non-goals and release gate live in the design doc and implementation plan; this folder is scoped to a single phase.

## Architecture decisions
- Vendor a version-pinned `dlpack.h` + a tiny C oracle under `tests/_abi_oracle/`; commit per-platform JSON snapshots for T0. ADR: n/a (design §7.1).

## Phase — DLPack ABI mirrors & compiled oracle
**Scope.** ctypes mirrors of `DLDevice`, `DLDataType`, `DLTensor`, `DLManagedTensor`, `DLPackVersion`, `DLManagedTensorVersioned`, flag constants; plus the vendored header + C oracle + snapshots.

**Steps.**
1. Author the ctypes structs: packed `DLDataType (uint8, uint8, uint16)` with no stray padding; `flags (uint64)` before `dl_tensor` in the versioned struct.
2. Vendor pinned `dlpack.h` + the C oracle emitting JSON; commit `linux-x86_64`, `linux-aarch64`, `macos-arm64`, `windows-x86_64` snapshots.

**Tests.**
- T1 compiled oracle (`sizeof`/`offsetof` vs ctypes; skip on T0) — the single most load-bearing test.
- T0 fallback vs committed JSON snapshots.
- Field-order regression via `memoryview` byte pattern.

**Exit criteria.** Gate 6 green; the compiled oracle is green on the T1 CI job.

**Public-API snapshot.** (no public export; internal `_dlpack/_abi`)

## Risks & open questions
- Struct layout drift from `dlpack.h` -> compiled oracle + committed snapshots.
- Unintended ctypes padding -> packed-struct assertion.
