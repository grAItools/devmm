# `devmm` — implementation plan for an autonomous coding agent

Companion to `devmm-design.md` (r2.2). The design document is the specification; this document is the build order, the verification machinery, and the rules of engagement. Section references (§) point at the design doc.

## 0. The agent contract

**On "guaranteed correct".** No process guarantees correctness in the absolute sense; what this plan does is make every requirement *executable*, so that "done" is defined as "all gates green" and the gates collectively encode the specification. The correctness strategy rests on four legs, all mandatory:

1. **Test-first per phase.** Every phase below lists tests that must be written and observed to fail (or skip) *before* the implementation they cover. The agent never writes implementation code for behavior that has no test.
2. **External oracles, not self-agreement.** Wherever a reference implementation exists, `devmm` is tested *against it*, never merely against its own expectations: NumPy for stride math and DLPack consumption, a **compiled `dlpack.h`** for ABI layout, `array-api-strict` for protocol conformance, rmm/CuPy/Numba behavior on GPU CI.
3. **Property-based invariants** (hypothesis) for everything algorithmic (layout resolution, adaptor accounting, registry semantics).
4. **Phase gates.** Each phase ends with a gate command that must pass before the next phase starts. Gates are cumulative: gate *N* runs gates 1..N. No skipping, no `xfail` left behind, no `# type: ignore` without a linked issue comment.

**Hard rules.**

- Public API is frozen by test: `tests/test_public_api.py` snapshots `devmm.__all__` and the signatures of everything in it (via `inspect.signature`). Changing the snapshot requires changing the design doc first.
- `mypy --strict` clean and `ruff check` + `ruff format --check` clean at every gate. `py.typed` ships from phase 0.
- Zero runtime dependencies in the core (`pyproject` `dependencies = []`) — enforced by a gate step that installs the wheel into a bare venv and runs the CPU test subset there. NumPy, hypothesis, etc. live in `[test]` extras only.
- Every module the design names exists with the exact path in §2 of the design doc. New helper modules are private (`_`-prefixed).
- Environment tiers: **T0** = any CPython ≥ 3.10, no compilers, no GPUs (phases 0–8 fully verifiable here except the compiled-ABI oracle); **T1** = T0 + a C compiler (enables the ABI oracle); **T2** = NVIDIA GPU + rmm/CuPy/Numba; **T3** = AMD GPU + ROCm. CI always runs T0+T1 (Linux/macOS/Windows matrix for T0; Linux for T1); T2/T3 jobs are required for the *final* release gate but individual phases marked `[GPU]` may land with their tests skipping on T0 — the tests must still be written, hardware-gated with `pytest.mark.gpu_cuda` / `gpu_rocm`.
- Commit per task, gate per phase. If a gate fails, fix forward within the phase; never disable a test to pass a gate.

**Repository tooling (fixed):** `uv` for environments and builds; `pytest` + `hypothesis` + `pytest-cov`; `ruff` (lint+format); `mypy --strict`; `just` as the command runner. `justfile` targets: `just lint`, `just typecheck`, `just test`, `just gate N`, `just gate-all`.

---

## Phase 0 — scaffold and CI skeleton

**Deliverables.** `pyproject.toml` (hatchling or flit backend; `requires-python >= 3.10`; extras: `test`, `cuda`, `rocm`, `cupy`, `numba`), full package tree from design §2 with empty modules + docstrings, `py.typed`, `justfile`, GitHub Actions workflow with the T0 matrix (3 OS × min/max Python) and a T1 Linux job, `tests/test_public_api.py` with an initially-empty snapshot, `tests/conftest.py` defining the `gpu_cuda`/`gpu_rocm` markers and a `recording_mr` fixture placeholder.

**Tests first.** `test_public_api.py` (empty snapshot passes), `test_packaging.py`: build wheel with `uv build`, install into a fresh venv with *no* other packages, `python -c "import devmm"` succeeds — this test is the zero-dependency enforcement and runs at every gate.

**Gate 0.** `just gate 0` = lint + typecheck + tests + wheel-in-bare-venv. **Traps:** don't let the build backend package `tests/`; ensure `__init__.py` re-exports nothing yet (snapshot is empty).

## Phase 1 — `_core/dtypes.py`, `_core/device.py`

**Deliverables.** `DType` (frozen, `(code, bits, lanes)`), named aliases, constructors from Array-API dtype strings and duck-typed NumPy dtypes (no numpy import in module scope); `DeviceType` IntEnum with DLPack codes, `Device` with `from_string`, `__dlpack_device__` (§3.1, §3.7).

**Tests first.**
- Table-driven: every alias ↔ exact `(code, bits, lanes)` from `dlpack.h` (`kDLInt=0, kDLUInt=1, kDLFloat=2, kDLBfloat=4, kDLComplex=5, kDLBool=6`), itemsize property.
- Differential oracle: for every alias with a NumPy counterpart, `DType.from_any(np.dtype(x)).itemsize == np.dtype(x).itemsize` and round-trips.
- `Device.from_string` fuzz: hypothesis over well-formed and malformed strings; parse-format round-trip; hashability/equality; `__dlpack_device__() == (int(type), index)`.
- Import hygiene: a subprocess test asserting `import devmm._core.dtypes` does not pull `numpy` into `sys.modules`.

**Gate 1.** All above + snapshot updated to export `DType`, `Device`, `DeviceType`.

## Phase 2 — `_core/layout.py`

**Deliverables.** `LayoutPolicy` ABC (upper-bound alignment properties), frozen `Layout` with `validate()` and `is_contiguous`, shared resolution helper, shipped policies `RowMajor`, `ColMajor`, `Permuted`, `Aligned`, `DeviceOptimal` (§3.6).

**Tests first.**
- **Injectivity oracle (the key correctness test):** hypothesis over shapes (ndim ≤ 5, extents ≤ 8) and all shipped policies — enumerate *every* index tuple, compute `offset = Σ idx[i]*strides[i]`, assert all offsets are distinct, non-negative, and `max_offset*itemsize + itemsize <= required_nbytes`. This single property rules out overlap and out-of-bounds by construction.
- Differential oracle vs NumPy: for `RowMajor`/`ColMajor` and non-degenerate shapes, `layout.strides == tuple(s // itemsize for s in np.empty(shape, dt, order).strides)` and `required_nbytes == np.empty(...).nbytes`.
- Alignment postconditions for `Aligned`: line-pitch divisibility, `required_nbytes % base_alignment == 0` where specified; padding is minimal (subtracting itemsize from the padded extent breaks divisibility).
- Upper-bound invariant: `layout.base_alignment <= policy.base_alignment` (and unit-stride analogue) across the hypothesis corpus; `layout.policy is policy`.
- Frozen/hashable: policies and layouts usable as dict keys; mutation raises.
- `validate()` rejects hand-built overlapping/negative-stride layouts (constructive counterexamples).
- Edge cases: ndim=0, zero extents, extents of 1, huge extents (overflow-safe `required_nbytes` — Python ints, but assert no float sneaks in).

**Gate 2.**

## Phase 3 — `_core/stream.py`, `_core/memory_resource.py`

**Deliverables.** `Stream` ABC + `CpuStream` no-op + sentinels; `DeviceMemoryResource` ABC with `stream_ordered`, `guaranteed_alignment()`, `available_memory()`; `CallbackMemoryResource`; `StatisticsAdaptor`, `LoggingAdaptor`, `LimitingAdaptor` (§3.2, §3.3).

**Tests first.**
- Build `testing/_recording.py` first: `RecordingMemoryResource` (deterministic fake pointers, logs every call, detects double-free/foreign-free/size-mismatch) — this is the fixture the rest of the suite lives on, so it gets its own tests.
- Adaptor forwarding: exact `(nbytes, stream)` pass-through; strong `upstream` chain survives `del` + `gc.collect()` (checked via `weakref` to the upstream).
- `StatisticsAdaptor` accounting under hypothesis-generated interleavings of alloc/free: `current == Σ live`, `peak == max prefix`, `total` monotone; thread soak (8 threads × 1k ops) ends balanced.
- `LimitingAdaptor`: boundary-exact (allocation that lands on the limit succeeds; +1 byte raises `MemoryError`; failed allocation does not count).
- `CallbackMemoryResource` invokes callbacks with correct arguments and propagates exceptions untouched.
- `CpuStream.synchronize()`/`wait_raw()` are no-ops; sentinel identity semantics.

**Gate 3.**

## Phase 4 — `mrs/cpu.py`: `BytearrayMemoryResource`, `MallocMemoryResource`

**Deliverables.** As per §5.1, including the Windows `_aligned_malloc`/`_aligned_free` branch and the family-tracking that prevents mismatched frees.

**Tests first (run against BOTH MRs via a parametrized fixture — this fixture becomes the reusable MR conformance suite, `devmm.testing.mr_conformance`).**
- Write-then-read oracle: `ctypes.memmove` a known pattern into the returned pointer, read it back byte-exact; adjacent allocations don't alias (fill A, fill B, verify A intact).
- Alignment: returned `ptr % requested_alignment == 0` across hypothesis sizes/alignments; `guaranteed_alignment()` honesty (MallocMR exact; BytearrayMR reports 1 and `empty()`-level over-allocation is tested later in phase 7).
- Bookkeeping: after N alloc/free pairs the MR's internal tables are empty (exposed via a `_debug_live_count()` testing hook); double-free raises; free of unknown pointer raises.
- BytearrayMR pinning: while allocated, resizing the backing bytearray raises `BufferError` (whitebox); after free, no references remain (weakref on the bytearray dies).
- MallocMR: subprocess leak canary — loop 10⁵ alloc/free of 1 MiB, assert RSS growth below threshold (Linux-only, best-effort marker `slow`).
- Zero-byte allocation contract (returns a ptr or 0, free round-trips) pinned down explicitly.

**Gate 4** includes the conformance suite passing for both CPU MRs on all three OSes.

## Phase 5 — `_core/buffer.py`, `_core/registry.py`

**Deliverables.** `DeviceBuffer` (§3.5) with `weakref.finalize` safety net, idempotent `free`, context manager, `copy_from_host`/`copy_to_host` (CPU path via `ctypes.memmove` for now; routed through the runtime from phase 8); registry with contextvars scoping (§3.4).

**Tests first.**
- Lifecycle against `RecordingMemoryResource`: `free()` calls `deallocate` exactly once with the allocation stream by default and with an explicit stream when given; second `free()` no-op; use-after-free raises.
- Finalizer: drop all refs, `gc.collect()`, recording MR shows exactly one deallocate; **cycle test**: embed the buffer in a reference cycle, collect, still exactly one deallocate; finalizer does not resurrect (`weakref` to buffer dies).
- Interpreter-shutdown safety: subprocess that allocates and exits without freeing → clean exit code, no traceback on stderr.
- Registry: strong-ref semantics (setting an MR, dropping the caller's ref, MR stays alive); `using_memory_resource` restores on exit and on exception; contextvar isolation across `threading.Thread` and `asyncio.gather` tasks (each sees its own override); lazy default raises cleanly before phase 8 wires runtimes in (temporary sentinel behavior, replaced later — test updated then, tracked in the phase-8 checklist).
- `copy_from_host`/`copy_to_host` byte-exact round-trip on CPU MRs.

**Gate 5.**

## Phase 6 — `_dlpack/_abi.py` + the compiled oracle

**Deliverables.** ctypes mirrors of `DLDevice`, `DLDataType`, `DLTensor`, `DLManagedTensor`, `DLPackVersion`, `DLManagedTensorVersioned`, flag constants; a **vendored, version-pinned `dlpack.h`** under `tests/_abi_oracle/` with a 30-line C program that prints `sizeof`/`offsetof` of every struct/field as JSON.

**Tests first.**
- T1 oracle test: compile the C program (`cc` discovered via `shutil.which`, test skips on T0), run it, compare every value against `ctypes.sizeof`/`ctypes.Structure.<field>.offset`. This is the single most load-bearing test in the project — a silent ABI mismatch corrupts memory in every consumer.
- T0 fallback: the same comparison against JSON snapshots checked in for `linux-x86_64`, `linux-aarch64`, `macos-arm64`, `windows-x86_64` (generated once on T1/CI and committed) so T0 platforms still verify layout.
- Field-order regression: constructing `DLManagedTensorVersioned` and setting every field by name round-trips through a `memoryview` byte pattern (catches accidental field reordering even if sizes coincide).

**Gate 6** requires the compiled oracle green on the T1 CI job. **Traps:** `DLDataType` is packed `(uint8, uint8, uint16)` — verify no unintended ctypes padding; `flags` is `uint64` *before* `dl_tensor` in the versioned struct.

## Phase 7 — `_dlpack/export.py`, `_core/tensor.py`, `empty()`/`empty_like()`

The critical phase; budget the most review here (§3.8, §7).

**Deliverables.** Capsule builder (single-block layout `[struct | shape | strides]`), module-global `CFUNCTYPE` deleter with `Py_IsInitialized` guard, `_Holder` incref/decref chain via `ctypes.pythonapi`, capsule destructor honoring the consumed-rename protocol, version negotiation, stream handoff hook (runtime-routed; CPU no-op now), `Tensor`, `empty`, `empty_like` with `guaranteed_alignment()`-aware over-allocation.

**Tests first.**
- **NumPy round-trip family (the primary oracle):** for hypothesis-generated (shape, dtype, policy) triples on CPU: write a pattern via `copy_from_host`, `a = np.from_dlpack(t)`, assert dtype/shape/values; **padded layouts**: `a.strides` equals the layout's strides in bytes and mutating `a` through NumPy round-trips back through `copy_to_host` (proves genuine zero-copy).
- Negotiation: `t.__dlpack__(max_version=(1, 1))` yields a capsule named `dltensor_versioned`; `max_version=None` yields `dltensor`; capsule names asserted via `ctypes.pythonapi.PyCapsule_IsValid`; `read_only=True` + legacy request → `BufferError`; NumPy consumes both paths (parametrize over NumPy's own negotiation by calling `np.from_dlpack` on wrapper objects that clamp `max_version`).
- **Deleter lifecycle (memory-safety core, all against `RecordingMemoryResource`):**
  (a) consume capsule → delete tensor/buffer refs → gc → *no* deallocate; delete the consumer array → gc → exactly one deallocate.
  (b) unconsumed capsule: create via `__dlpack__`, delete the capsule → holder released, and buffer freed once user refs are gone.
  (c) double-export: two capsules from one tensor, interleaved consumer lifetimes → one deallocate, at the correct time.
  (d) refleak harness: run (a) 10⁴ times, assert `sys.getrefcount`/`tracemalloc` deltas bounded; `gc.garbage` empty.
  (e) foreign-thread deleter: consume in NumPy, drop the array from a non-main thread — no deadlock, GIL correctly acquired (this is the ctypes-callback property the design relies on).
  (f) shutdown: subprocess where a consumer array outlives module teardown — clean exit.
- Protocol refusals: mismatched `dl_device` → `BufferError`; `copy=True` → `BufferError` (v1); zero-size tensor exports `data == NULL` and NumPy accepts it; `stream=-1` accepted, invalid CUDA stream ints rejected by validation logic (unit-tested against the validation table, no GPU needed).
- `empty_like` duck-typing against NumPy arrays and `array-api-strict` arrays.
- `array-api-strict.from_dlpack` consumes `devmm` tensors (conformance oracle #2).
- Public-API snapshot grows: `Tensor`, `empty`, `empty_like`.

**Gate 7** additionally runs the refleak harness and the subprocess tests on all three OSes. **Traps:** keep the `CFUNCTYPE` object referenced at module scope *and* verify by test that the deleter survives `gc.collect()`; `PyCapsule_New`'s destructor must check the current capsule name before running the managed deleter; incref/decref must go through `ctypes.pythonapi.Py_IncRef/Py_DecRef` with `argtypes = [ctypes.py_object]` set explicitly.

## Phase 8 — `_runtimes/base.py`, `_discovery.py`, `_runtimes/cpu.py`

**Deliverables.** `DeviceRuntime` protocol, `(name, probe, loader)` registry, entry-point loading (`devmm.runtimes` group), `available_runtimes()` / `runtime_names()` / `runtime_for()`, `DEVMM_RUNTIME` override, CPU runtime (memcpy via `ctypes.memmove`, default MR = `MallocMemoryResource`); registry lazy default now routes through `runtime_for(device).default_memory_resource(device)` (update the phase-5 sentinel test).

**Tests first.**
- Probe laziness: `runtime_names()` leaves heavyweight modules out of `sys.modules` (subprocess assertion); `available_runtimes()` constructs only passing probes.
- Fake third-party runtime registered via an in-test entry point (using `importlib.metadata` shims) is discovered, ordered after built-ins, and loadable.
- `DEVMM_RUNTIME=cpu` forces selection; bogus value → `RuntimeUnavailableError` with actionable message.
- End-to-end on CPU: `empty(..., device=Device.from_string("cpu"))` with no explicit MR uses the runtime default; the full phase-7 round-trip suite re-runs through this path.

**Gate 8.** At this point the library is *feature-complete for CPU* and the whole DLPack surface is oracle-verified. Tag `v0.1.0a1`.

## Phase 9 — CUDA `[GPU]`: `_runtimes/cuda.py`, `mrs/cuda.py` (`CudaRuntimeMR`, `RmmMR`)

**Design-for-testability requirement (mandatory):** the CUDA runtime and `CudaRuntimeMemoryResource` take an injected `api` object encapsulating every `libcudart` call (loaded via ctypes in production). All control-flow logic — error-code → exception mapping, async-path selection, event record/wait sequencing in `make_stream_wait`, device activation push/pop — is unit-tested on T0 against a scripted `FakeCudartApi` that records calls and can inject failures. Only the thin `api` construction itself is GPU-only.

**Tests first.**
- T0 (fake api): allocate/free call sequences byte-for-byte as expected; `async_alloc="auto"` selects `cudaMallocAsync` iff the fake reports driver support; every nonzero status maps to the right exception with the CUDA error string; `make_stream_wait` emits create-event → record(producer) → stream-wait(consumer) → destroy-event; `activate_device` restores the previous device on exception.
- T0: `RmmMemoryResource` unit-tested against a fake exposing rmm's Python MR signature (`allocate(nbytes, stream)`), stream translation via `__cuda_stream__`, strong ref to the wrapped MR.
- T2 (`gpu_cuda`): the phase-4 MR conformance suite over `CudaRuntimeMR` (sync and async) and `RmmMR(rmm.mr.CudaMemoryResource())` (write/read via `cudaMemcpy` through the runtime's `memcpy`); full DLPack round-trips through **CuPy and PyTorch** (`cp.from_dlpack`, `torch.from_dlpack`), padded strides included; the **stream-race canary** from design §9: producer writes on stream A via a CuPy kernel, consumer imports with `__dlpack__(stream=B)` — correct with handoff enabled, and the test demonstrates it can fail with handoff forcibly disabled (guarded, best-effort); rmm pool statistics agree with `StatisticsAdaptor` counts.

**Gate 9.** T0 portion green everywhere; T2 job green on the CUDA runner.

## Phase 10 — ROCm `[GPU]`: `_runtimes/rocm.py`, `mrs/rocm.py`

**Deliverables.** Shared FFI shim (`_runtimes/_gpulib.py`) factored out of phase 9 so CUDA/ROCm differ only in symbol names and error tables; `HipRuntimeMR`, `HipmmMR`; the platform-keyed probe + `rmm`-module disambiguation (§4.2).

**Tests first.** T0: fake-api suite re-parametrized for HIP (this is where the shared shim pays off — same tests, second symbol table); disambiguation unit tests: fake `rmm` modules with CUDA-ish vs HIP-ish markers, probe picks correctly, `DEVMM_RUNTIME` override wins. T3: conformance + round-trip suite (CuPy-ROCm and/or PyTorch-ROCm as available on the runner).

**Gate 10.** T3 marked *advisory* until AMD CI hardware is provisioned; release gate (phase 12) decides its status explicitly.

## Phase 11 — third-party MRs and `integrations/` `[GPU for some]`

Order within the phase: (a) `CupyAllocatorMR` + `integrations/cupy.py`; (b) `integrations/numba.py` EMM plugin; (c) `NumpyHandlerMR` + `integrations/numpy.py`; (d) `integrations/rmm.py` bridges.

**Tests first.**
- Cycle detection is T0-testable with fakes and comes first: `integrations.cupy.install(CupyAllocatorMR(...))` raises; every `install()` returns an uninstaller that restores prior state (asserted via the fake's hook registry) including under exceptions (context-manager form).
- (c) is the delicate one and is CPU-only, so it gets the full treatment: ctypes mirror of `PyDataMem_Handler` verified against a NumPy-version-parametrized offset table; `install(StatisticsAdaptor(MallocMR(...)))` → allocate NumPy arrays → statistics grew; `uninstall()` → `np.core.multiarray.get_handler_name()` (or 2.x equivalent) reports the prior handler; arrays allocated *during* installation remain freeable *after* uninstall (NumPy carries the handler per-array — test pins this); supported-NumPy-range guard raises cleanly outside the pinned range.
- (a) T2: allocation observed inside the configured CuPy pool (`pool.used_bytes()` delta), `deallocate` returns memory to the pool; current-device/current-stream context correctness by allocating on a non-default stream and asserting via pool introspection.
- (b) T2: run Numba's own EMM plugin test protocol (numba exposes hooks); a `@cuda.jit` kernel writes into memory that `devmm` statistics account for.

**Gate 11.**

## Phase 12 — conformance suite as a public API, docs, release gate

**Deliverables.** `devmm.testing.mr_conformance(mr_factory)` and `devmm.testing.dlpack_conformance(device)` promoted to documented public API (so third-party MR/runtime authors get the same guarantees); README + API docs (mkdocs-material) with every code block executed by `pytest --doctest-glob` or `mkdocs-doctest`; CHANGELOG; version `0.1.0`.

**Release gate (the final correctness checklist, all must be green):**

1. `just gate-all` on the full T0 matrix (3 OS × min/max CPython) and T1.
2. T2 GPU job: full suite including stream-race canary and torch/cupy round-trips.
3. T3 job green, or an explicit signed-off waiver documenting which ROCm tests ran manually and where.
4. Coverage ≥ 95% on `_core` + `_dlpack`, ≥ 90% overall (T0-measurable code); every uncovered line annotated `# pragma: no cover` with a reason.
5. Refleak harness and subprocess shutdown tests green under `PYTHONDEVMODE=1` and `faulthandler` enabled.
6. Optional-but-recommended job: the phase-7 suite on a `--with-pydebug` CPython build (refcount assertions catch incref/decref bugs ctypes can mask).
7. Public-API snapshot matches design doc §2/§3 exactly; `mypy --strict` clean; wheel-in-bare-venv test green.
8. Design-doc conformance review: a final pass mapping every "must"/"raises"/"contract" sentence in `devmm-design.md` to at least one test ID — the traceability table is committed as `tests/traceability.md`. Any unmapped sentence is either tested or the design doc is amended, before tagging.

---

## Risk register → mitigating tests (quick index)

| Risk | Phase | Mitigation test |
|---|---|---|
| ctypes struct layout drifts from `dlpack.h` | 6 | compiled oracle + committed per-platform snapshots |
| CFUNCTYPE deleter GC'd → segfault | 7 | deleter-survives-gc test + refleak harness + pydebug job |
| Use-after-free across capsule/consumer lifetimes | 7 | lifecycle matrix (a)–(f) on RecordingMR |
| Stream handoff wrong ordering | 9 | fake-api sequence assertions + GPU race canary |
| `_aligned_malloc`/`free` family mismatch (Windows) | 4 | family-tracking test on Windows CI leg |
| hipMM `rmm` module ambiguity | 10 | probe disambiguation unit tests + env override |
| NEP-49 ABI drift across NumPy versions | 11 | version-parametrized offset table + range guard |
| Global-state corruption by `install()` | 11 | restore-on-uninstall + exception-path tests |
| Accidental runtime dependency creep | all | wheel-in-bare-venv gate step |
| Silent public-API drift | all | signature snapshot test |

## Suggested agent task granularity

One PR-sized task ≈ one bullet group above. Within a phase: (1) write the listed tests, run, confirm red/skip; (2) implement the minimum to green; (3) refactor with tests green; (4) run `just gate N`; (5) update `tests/traceability.md` incrementally rather than at the end. Estimated ordering-critical path: 0 → 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 (CPU-complete) → {9, then 10} ∥ 11(c) → 11(a,b,d) → 12.
