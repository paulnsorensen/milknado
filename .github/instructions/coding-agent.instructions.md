---
applyTo: "**"
excludeAgent: "code-review"
---

## Coding Agent Guidelines

When implementing changes:

- Read existing code before modifying it — understand the patterns in use
- Follow the existing code style of the file you are editing
- Keep changes minimal and focused on the issue at hand
- Do not refactor surrounding code unless the issue requires it
- Do not add docstrings to helper functions or private methods with clear names
- Do not introduce new dependencies without explicit approval
- Write tests for new functionality — match the existing test patterns in `tests/`
- Prefer editing existing files over creating new ones
- Run `just check-llm` before opening a PR — THE gate: lint, format, file-size, import contracts, tests, and project + diff coverage at 95%. It is non-mutating; if it fails on lint/format, run `just lint-fix` and re-run the gate

## Architecture Rules

- New domain concepts go in their own module under `src/milknado/domains/`
- Public API is exposed through `__init__.py` barrel files only
- Do not create abstract classes, interfaces, or factories unless there are 2+ concrete implementations
- Core models must not import infrastructure code
- Cross-slice translation belongs in an explicit bridge module, not inside the pure slice
- No migration code — this project is pre-release; change the code directly instead of adding backfills, deprecation shims, or compatibility layers
- A new underscore-private module inside a domain must be added to the "boundary modules must use domain barrels" import-linter contract in `pyproject.toml`
