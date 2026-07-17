# devmm

**Device Memory Manager** — a uniform, pure-Python interface for allocating,
deallocating and managing device memory across **CPU, CUDA and ROCm**. It wraps
existing allocators (rmm, hipMM, CuPy pools, libc) rather than implementing
allocation strategies, and exposes every allocation as a zero-copy
**DLPack ≥ 1.0** producer consumable by any Array-API library (NumPy, CuPy,
PyTorch, JAX). Zero required dependencies; `ctypes` only.

The authoritative design is [`work/devmm-design.md`](work/devmm-design.md);
the API reference with executable examples is [`docs/api.md`](docs/api.md).

## Usage

Allocate with a layout policy, consume zero-copy from any DLPack consumer
(the examples below are doctests, executed by the test suite):

```python
>>> import numpy as np
>>> import devmm
>>> t = devmm.empty((2, 3), "float32", layout=devmm.RowMajor())
>>> t.shape, t.strides
((2, 3), (3, 1))
>>> a = np.from_dlpack(t)  # zero-copy view over the devmm allocation
>>> a[...] = 1.0
>>> float(a.sum())
6.0

```

Layout control at allocation time — pad the innermost extent so every row
starts on a 128-byte boundary; consumers see the padded strides:

```python
>>> padded = devmm.empty(
...     (4, 3),
...     "float32",
...     layout=devmm.Aligned(devmm.RowMajor(), unit_stride_alignment=128, base_alignment=256),
... )
>>> padded.strides  # elements: 32 float32 per 128-byte line
(32, 1)
>>> np.from_dlpack(padded).strides  # bytes
(128, 4)

```

Route allocations through an adaptor stack and the per-device registry:

```python
>>> from devmm.mrs.cpu import MallocMemoryResource
>>> stats = devmm.StatisticsAdaptor(MallocMemoryResource())
>>> with devmm.using_memory_resource(stats):
...     tracked = devmm.empty((256,), "float32", layout=devmm.RowMajor())
>>> stats.current_bytes, stats.peak_bytes
(1024, 1024)
>>> tracked.buffer.free()
>>> stats.current_bytes
0

```

## Quickstart

```sh
make test    # fast unit tests
make lint    # static checks
make fmt     # auto-format
make verify  # full gate; what the agent runs before claiming done
```

## AI coding agents

This repository follows the **agent-agnostic harness** convention:

- `AGENTS.md` is the canonical instruction file for every agent
  (Codex, OpenCode, Cursor, Amp, Factory, Gemini CLI, Copilot, …).
- `CLAUDE.md` is a one-line import of `AGENTS.md` plus Claude-Code-only
  stanzas (skills, subagents, slash commands, hooks).
- Shared agent assets (skills, subagents, and slash commands) live under
  `.agents/` and are symlinked into `.claude/` and `.opencode/`.
- Per-feature specs go under `work/<YYYY-MM>-<slug>/`.
- Architecture decisions are in `docs/adr/` (Michael Nygard format, append-only).

See `AGENTS.md` for the full conventions.

## License

BSD-3-Clause
