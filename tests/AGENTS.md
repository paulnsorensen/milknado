# Tests

These rules add to the root `AGENTS.md` for files under this directory.

## What to test

- Test behavior through the owning seam: the Executor, a port, the CLI runner, or the registered MCP transport. Do not assert only on a private helper.
- Do not fake the unit under test. Use fakes only for ports and external systems.
- Test MCP tools with `meta/mcp_contracts.py` (`call_tool`, `registered_tools`).
- Patch a name in the module that looks it up, not in the module that defines it.
- Do not skip, xfail, or lengthen a timeout to pass an intermittent test. Reproduce it and find the cause.

## Environment

- The autouse `_isolate_worker_identity` fixture in `conftest.py` removes every `MILKNADO_*` variable. Set the identity variables that a test needs explicitly.
- Browser tests live in `browser/`, carry the `browser` marker, and run over the committed build in `../src/milknado/web/static/`.

## Lint and type exceptions (narrows root "Code Style")

- `../pyproject.toml` relaxes ruff ANN, ARG, BLE, B008, B017, PTH, and SIM for `tests/**`. All other ruff rules apply.
- `just typecheck` covers this directory and fails on warnings, including private-name use. Give a shared test helper a public name.
