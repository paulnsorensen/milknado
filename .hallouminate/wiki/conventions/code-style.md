# Code Style & Engineering Principles

How code in milknado must look and be structured. Source: `AGENTS.md` (`CLAUDE.md`
symlinks to it) and `pyproject.toml`.

## Language & formatting

- **Python 3.11+** — `requires-python = ">=3.11"`, `target-version = "py311"` (`pyproject.toml`).
- **ruff** is the formatter and linter; **line length 99** (`[tool.ruff] line-length = 99`).
  Lint rule sets: `E`, `F`, `I`, `UP` (`[tool.ruff.lint] select`). `just lint-fix`
  applies both autofix and format; the gate (`just check-llm`) only checks.

## Complexity budgets

Keep units small so they stay readable and testable:

- Max function: **40 lines**
- Max file: **300 lines**
- Max params: **4**

When a function or file pushes these limits, split it — the budget is a signal that
a slice or helper wants to be extracted, not a number to game.

## Naming

- `snake_case` — functions
- `PascalCase` — classes
- `SCREAMING_SNAKE_CASE` — constants
- `kebab-case` — file names

Name things after **business concepts**, not technical abstractions (principle 5).

## Architecture — Sliced Bread

Vertical slices, not layered tech buckets:

- Domain slices live under `src/milknado/domains/`.
- Infrastructure adapters live under `src/milknado/adapters/`.

Entry points: `milknado` CLI (`src/milknado/cli/`) and the `milknado-mcp` MCP
server (`src/milknado/mcp/server.py`). Keep business logic in slices and out of the
adapters/entry points (principle 3 — separate business logic from infrastructure).

## The six engineering principles

1. **Trust nothing from external sources** — validate at boundaries.
2. **Fail fast and loud** — no silent failures.
3. **Separate business logic from infrastructure.**
4. **YAGNI** — build only what's needed now; no premature abstraction.
5. **Name things after business concepts**, not technical abstractions.
6. **Minimize state mutation** — prefer immutable patterns for predictable behavior.

## No Migration Code

Milknado is **pre-release**. Do **not** add migration backfills, deprecation shims,
or backward-compatibility layers. When a shape changes, change it outright — there is
no old data or old API to preserve, so compat scaffolding is pure liability.
