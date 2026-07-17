.PHONY: help test test-all lint typecheck fmt fmt-check verify gate-all dev clean

help:  ## Show this help
	@awk 'BEGIN {FS = ":.*##"; printf "Usage: make \033[36m<target>\033[0m\n\nTargets:\n"} \
	/^[a-zA-Z0-9_-]+:.*?##/ { printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

# Tests run with the `test` extra (numpy, array-api-strict): the suite's
# differential oracles and DLPack round-trips need a consumer library (§9).
test:  ## Run fast unit tests
	uv run --extra test pytest -q

test-all: test  ## Run the full suite (override to add integration/e2e)
	@echo "test-all: extend this target with integration suites as needed"

lint:  ## Run static checks (does not auto-fix)
	uv run ruff check .

typecheck:  ## Strict static types (mypy config lives in pyproject.toml)
	uv run mypy

fmt:  ## Auto-format: apply ruff lint fixes (imports, etc.) then format
	uv run ruff check --fix .
	uv run ruff format .

fmt-check:  ## Check formatting without modifying files
	uv run ruff format --check .

verify:  ## What the agent runs before claiming done
	@./scripts/verify.sh

# Gates are cumulative, so every gate-N aliases the current full verify gate
# (see docs/adr/0002-task-runner-make-over-just.md).
gate-%: verify
	@echo "gate-$*: green"

gate-all: verify  ## Cumulative gate: lint + mypy --strict + tests + packaging
	@echo "gate-all: green"

dev:  ## Run the local dev workflow (override per-project)
	@echo "dev: override this target to start your dev server / watcher"

clean:  ## Remove generated artefacts (override per-project)
	@echo "clean: override this target to remove build/ dist/ etc."
