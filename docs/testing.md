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

## Running the CUDA GPU suite on hardware

The T2 suites (`tests/test_cuda_gpu.py`, `tests/test_integrations_gpu.py`) are
skip-gated behind the `gpu_cuda` marker and opt in with `DEVMM_GPU=cuda`
(`tests/conftest.py`). On a CUDA-12 host:

1. Install the consumer stack — the `gpu-test-cuda` extra plus a PyTorch CUDA
   wheel (PyTorch's CUDA builds, including the aarch64/sbsa ones, live on its
   own index, not PyPI):

   ```sh
   uv pip install '.[test,gpu-test-cuda]'
   uv pip install torch --index-url https://download.pytorch.org/whl/cu128
   ```

2. Run the suite (or `make test-gpu-cuda`, which wraps this):

   ```sh
   env -u CUDA_HOME -u CUDA_PATH DEVMM_GPU=cuda uv run --no-sync \
       pytest tests/test_cuda_gpu.py tests/test_integrations_gpu.py
   ```

Two hardware-only gotchas, both about Numba's JIT toolchain rather than devmm:

- **libnvvm ↔ nvjitlink alignment.** Numba compiles a kernel with `libnvvm`
  and links it with `nvjitlink`; if `libnvvm` is newer than `nvjitlink`, the
  link fails with `nvJitLinkError: ERROR 4 in nvvmAddNVVMContainerToProgram,
  may need newer version of nvJitLink library`. The `gpu-test-cuda` extra keeps
  `nvidia-cuda-nvcc-cu12` (libnvvm) in the 12.8 series so it stays minor-aligned
  with the nvjitlink PyTorch's wheels pin.
- **A newer system CUDA shadows the wheels.** If `CUDA_HOME` points at a system
  CUDA newer than the cu12 wheels (e.g. a 13.x module on an HPC box), Numba
  picks up that `libnvvm` and the alignment above breaks again. Clearing
  `CUDA_HOME`/`CUDA_PATH` (as the run command and `make test-gpu-cuda` do) makes
  Numba use the cu12 wheels instead.

## Coverage targets

The behavioural bar comes first: every shipped `LayoutPolicy`, every CPU MR,
and **both** DLPack capsule variants (versioned + legacy) have round-trip
coverage, and every `__dlpack__` refusal path (the `BufferError` cases) is
asserted. GPU paths are covered by the fake-api suites on CPU CI and by the
smoke jobs on GPU CI.

The release gate adds numeric thresholds (`make coverage`, enforced by the
`gate-coverage` CI leg on Linux): **≥ 90% overall** and **≥ 95% on
`devmm._core` + `devmm._dlpack`**. Anything left uncovered must carry a
reasoned `# pragma: no cover` or match a documented exclusion in
`pyproject.toml` (`[tool.coverage.report]`) — platform-only branches, debug
`__repr__`s, abstract seams. Never add a pragma to dodge writing a feasible
test.

The release gate also runs the whole suite under `PYTHONDEVMODE=1` with
`faulthandler` enabled (`make test-devmode`, the `gate-devmode` CI leg), so
the refleak harness and the shutdown subprocess tests execute with dev-mode
allocator and warning checks on.

Docs are executed as tests: every Python code block in `README.md` and
`docs/api.md` is a doctest session run by `tests/test_docs.py`.

## Determinism

- Time, randomness, and I/O must be injectable.
- Snapshot tests are fine but commit the fixture, not the snapshot run output.
- Flaky tests are bugs; quarantine them in a separate target and open an issue,
  don't `@retry`.
