# Plan — Phase 0 — Scaffold delta & CI skeleton

> **Context.** Step 00 of the devmm v0.1 build. Authoritative spec: [`work/devmm-design.md`](../devmm-design.md); build order: [`work/devmm-implementation-plan.md`](../devmm-implementation-plan.md) (Phase 0). Design sections: §2. This is the first step; no prior step is required.
> The overall v0.1 goals, non-goals and release gate live in the design doc and implementation plan; this folder is scoped to a single phase.

## Architecture decisions
- **Command runner: `make`, not `just`** — the implementation plan names `just`;
  this repo is standardized on `make` (verify gate, agent hooks and the
  four-phase loop all invoke `make verify`). `just gate N` maps to `make gate-N`.
  ADR: **ADR needed: task-runner `make` over `just` (harness alignment)**.
- **Python floor `>=3.12`** (plan says `>=3.10`) — keep the repo's existing pin;
  CI T0 matrix min=3.12. ADR: n/a.
- **Build backend `hatchling`**, `packages = ["devmm"]`, `py.typed` ships. ADR: n/a.
- **Zero-dependency enforcement:** a gate step builds the wheel, installs it into
  a bare venv, and imports it. ADR: n/a (design §8).
- **`mypy --strict` + `pytest-cov`** join the dev group; strict types clean at
  every gate, coverage thresholds enforced at the release gate. ADR: n/a.
- **External oracles:** NumPy, a compiled `dlpack.h`, `array-api-strict`, and
  GPU libraries on GPU CI. ADR: n/a (design §9).
- **Public API frozen by snapshot test**; changing it requires a design-doc
  change first. ADR: n/a.

## Phase — Scaffold delta & CI skeleton
**Scope.** Bring the existing scaffold up to the implementation-plan baseline: gate tooling, CI, strict typing, coverage, and the two always-on enforcement tests. Reuse the current package tree, `pyproject.toml` and `py.typed` — this is a delta, not a greenfield scaffold.

**Steps.**
1. Add `mypy` + `pytest-cov` to the dev group; add `[tool.mypy] strict = true`; keep the `test` extra = `numpy`, `array-api-strict`.
2. Add `make gate-N` / `make gate-all` targets wrapping lint + `mypy --strict` + tests + wheel-in-bare-venv; keep `make verify` as the hook-invoked cumulative gate.
3. Add a GitHub Actions workflow: T0 matrix (3 OS x {3.12, latest}) + a T1 Linux job.
4. Ensure the build backend does not package `tests/`.

**Tests.**
- `tests/test_public_api.py` — snapshot of `devmm.__all__` (initially empty) + member signatures; green while empty.
- `tests/test_packaging.py` — `uv build`, install into a fresh empty venv, `python -c "import devmm"` succeeds (zero-dependency gate step; runs at every gate).
- `tests/conftest.py` — `gpu_cuda`/`gpu_rocm` markers + a `recording_mr` fixture placeholder.

**Exit criteria.** Gate 0 green on the T0 matrix + T1; `__init__.py` re-exports nothing (empty snapshot).

**Public-API snapshot.** (unchanged; export surface stays empty)

## Risks & open questions
- Runtime-dependency creep -> the wheel-in-bare-venv gate step.
- Silent public-API drift -> the signature snapshot test.
- Build backend packaging `tests/` -> explicit wheel-contents check.
