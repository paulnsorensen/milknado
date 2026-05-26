# Milknado — Agent Instructions

## Build Gate

**Run `just build` before opening any PR.** It must pass cleanly.

```
just build   # lint-fix → test → coverage check (90% threshold, autofixes lint)
```

Output is compacted: each step prints `✓ <step>` on success and the full tool
output (with file:line) only on failure, then aborts. `just ci` is the same gate
with no autofixes — exactly what CI runs.

If `just build` is red, do not open a PR. Fix failing tests or coverage gaps first.
Lint errors are auto-fixed by `just build` — re-run after if files changed.

## Key Recipes

```bash
just install        # Install all dependencies (uv sync)
just lint           # Ruff check + format check (no changes — used by CI)
just lint-fix       # Ruff check + format with autofix
just test           # Run pytest (supports args: just test -k pattern)
just test-file <f>  # Run a single test file
just build          # Full pipeline with autofix — use this before every PR
just ci             # Full pipeline no autofix — CI uses this
just mcp-dev        # Run milknado-mcp under watchfiles, restarting on src/ changes
just clean          # Remove build artifacts and caches
```

`just mcp-dev` watches `src/milknado/` and restarts the MCP server on every change.
Because the server speaks stdio, a restart drops the existing connection — the client
must `/mcp` reconnect after each restart to pick up the new process.

## Project Overview

Milknado is a Mikado execution engine — it decomposes goals into dependency graphs and executes them as parallel ralph loops.

- **Entry points**: `milknado` CLI (`src/milknado/cli.py`), `milknado-mcp` MCP server (`src/milknado/mcp_server.py`)
- **Architecture**: Sliced Bread — vertical slices under `src/milknado/domains/`, adapters in `src/milknado/adapters/`
- **Tests**: `tests/` — pytest, 90% coverage required

## Code Style

- Python 3.11+, formatted with ruff (line length 99)
- Max function: 40 lines, max file: 300 lines, max params: 4
- snake_case functions, PascalCase classes, SCREAMING_SNAKE_CASE constants, kebab-case files

## Engineering Principles

1. Trust nothing from external sources (validate at boundaries)
2. Fail fast and loud — no silent failures
3. Separate business logic from infrastructure
4. YAGNI — only what's needed now
5. Name things after business concepts, not technical abstractions
6. Minimize state mutation

## No Migration Code

This project is pre-release. Do not add migration backfills, deprecation shims, or compatibility layers.
