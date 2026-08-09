# Copilot Instructions

## Engineering Principles

1. **Input Validation** - Trust nothing from external sources. Validate at system boundaries (user input, external APIs, file I/O). Internal code trusts internal code.
2. **Fail Fast and Loud** - Handle errors where they occur. No silent failures, no swallowed exceptions, no empty catch blocks. If something fails, the caller should know immediately.
3. **Loose Coupling** - Separate business logic from infrastructure. Core models and domain logic must not import HTTP frameworks, ORMs, or I/O libraries. Use dependency injection or protocols/interfaces at boundaries.
4. **YAGNI** - Build only what is needed now. No abstract base classes with one implementation, no plugin systems with one plugin, no configuration options that are never varied. If it is needed later, it can be written later.
5. **Real-World Models** - Name things after business concepts, not technical abstractions. `Order`, not `DataProcessor`. `PricingRule`, not `StrategyHandler`.
6. **Immutable Patterns** - Minimize state mutation. Prefer pure functions, return new values instead of mutating arguments, use immutable data structures where the language supports them.

## Complexity Budget

- **Functions**: Maximum 40 lines
- **Files**: Maximum 300 lines
- **Parameters**: Maximum 4 per function
- **Nesting**: Maximum 3 levels deep

If a function or file exceeds these limits, decompose it.

## Code Style

- **Classes**: PascalCase
- **Functions**: snake_case
- **Constants**: SCREAMING_SNAKE_CASE
- **Files**: kebab-case (or snake_case for Python modules, matching the existing tree)
- **Commits**: Conventional Commits format (`feat:`, `fix:`, `chore:`, etc.)

## Architecture

Follow vertical slice architecture:

- Each domain concept gets its own module/directory (see `src/milknado/domains/`)
- Public API is exposed through an `__init__.py` barrel file only
- Do not reach into another slice's internals — import from its public API
- Core models stay pure: no ORM decorators, no framework imports, no I/O
- `common/` is a leaf — it imports nothing from sibling domains
- One-directional dependencies only; cross-slice translation lives in an explicit bridge module (e.g. `domains/planning/batching_bridge.py`), never inside the pure slice
- Slice-boundary and layering contracts are enforced mechanically by import-linter (`lint-imports`, part of `just check-llm`); see `[tool.importlinter]` in `pyproject.toml`

**Growth pattern:**

1. Start with one file per concept
2. Extract a sibling file when the original gets crowded
3. When a file needs helpers, it becomes a facade with a subdirectory

## What NOT to Do

- Do not add docstrings to private methods or small helpers with clear names
- Do not create abstract base classes, factories, or registries unless there are multiple concrete implementations today
- Do not add error handling for conditions that cannot occur in the current system
- Do not add backwards-compatibility shims — change the code directly
- Do not wrap functions that add no logic — call the original directly
- Do not add type annotations to every local variable — annotate function signatures and let inference handle the rest

## No Migration Code

This project is pre-release. Do not add migration backfills, deprecation shims, or compatibility layers.

## Tech Stack

- **Language**: Python (`requires-python >= 3.11`)
- **Package/Env manager**: `uv` (lockfile: `uv.lock`)
- **Test framework**: pytest (with `pytest-cov`)
- **Lint + format**: Ruff (`--preview`)
- **Task runner**: `just`
- **Key libraries**: fastmcp, ortools (CP-SAT), typer, textual, tiktoken, pyyaml, rich, msgspec, code-review-graph
- **Shape**: MCP server (`src/milknado/mcp/server.py`) and `milknado` CLI over domain slices under `src/milknado/domains/`

## Build, Test, and Lint Commands

- **THE PR gate**: `just check-llm` — lint + format + file-size + import contracts + tests + project & diff coverage (95%, mirrors `codecov/patch`). Non-mutating; quiet on success. Green means open the PR, red means fix and re-run.
- Lint autofix + format: `just lint-fix` (run when `check-llm` fails on lint/format, then re-run the gate)
- Tests: `just test` (or `just test-file <path>`)
- Full autofix pipeline: `just build`; CI runs `just build-ci`
- Install deps: `just install`
