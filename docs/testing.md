# Testing strategy

## What the agent runs

- **Pre-claim-done gate**: `make verify` (= `scripts/verify.sh`).
- **Fast loop**: `make test` — must finish in <60s. Add slow suites under
  `make test-all`.

## Layering

The CPU memory resources make the **whole DLPack protocol testable without a
GPU** — that is the backbone of the suite (design §9).

- **Unit / protocol (no GPU)** — `numpy.from_dlpack(devmm.empty(...))`
  round-trips exercise capsule construction, versioned/legacy negotiation,
  deleter invocation (checked via `weakref` + `gc`), read-only flags, and
  padded-stride imports. Run against **both** `BytearrayMR` and `MallocMR`.
  Live in `tests/`.
- **Property-based** (`hypothesis`) — every shipped `LayoutPolicy`: strides are a
  valid permutation-derived set, `required_nbytes` bounds every addressable
  element, alignment postconditions hold, and `layout.base_alignment <=
  policy.base_alignment` (the §3.6 upper-bound invariant).
- **Mock-runtime** — `testing.MockRuntime` + a recording MR assert
  stream-ordering contracts (alloc/dealloc stream pairing, handoff event
  sequencing) without hardware.
- **Integration** (gated on optional-dep availability) — NEP-49 install/uninstall
  restores the prior handler; `CupyAllocatorMR` returns memory to its pool; the
  Numba EMM plugin passes Numba's EMM hooks.
- **ABI** — `ctypes.sizeof`/offset assertions on the `dlpack.h` and
  `PyDataMem_Handler` mirrors, guarding against silent field drift.
- **GPU CI** (only where hardware exists; outside the fast loop) — one smoke job
  per platform: rmm-pool + torch/cupy `from_dlpack` round trips and a
  stream-race canary.

## Coverage targets

No numeric gate. The bar is behavioural: every shipped `LayoutPolicy`, every CPU
MR, and **both** DLPack capsule variants (versioned + legacy) have round-trip
coverage, and every `__dlpack__` refusal path (the `BufferError` cases) is
asserted. GPU paths are covered by the mock runtime on CPU CI and by the smoke
jobs on GPU CI.

## Determinism

- Time, randomness, and I/O must be injectable.
- Snapshot tests are fine but commit the fixture, not the snapshot run output.
- Flaky tests are bugs; quarantine them in a separate target and open an issue,
  don't `@retry`.
