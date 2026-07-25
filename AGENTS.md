# Milknado — Agent Instructions

## Mandatory Python Workflow

For every Python implementation, refactor, review, or test change, invoke and follow the
local `$pythonic` skill at `.agents/skills/pythonic/SKILL.md`. Its Python de-slop and
Sliced Bread rules are required project conventions, not optional guidance.

Keep `CLAUDE.md` as a pointer to this file so every agent follows one authoritative set
of repository instructions.

## Build Gate

**`just check-llm` is the gate. Run it before opening any PR — and only this command.**

```
just check-llm   # lint + format + tests + project coverage + diff coverage (95%)
```

It is token-efficient by design: on success it prints a single PASS line; on failure
it prints only the failing step's output. Do not run the individual recipes to gate —
`check-llm` covers lint, format, tests, project coverage, and **diff coverage** in one
shot. Diff coverage mirrors `codecov/patch`: it fails if the lines this branch changes
(vs `origin/main`) aren't covered to 95% — catching a CI patch failure before you push.

- **Green** (PASS line printed) → open the PR.
- **Red** → do not open a PR. Fix the reported failure, then re-run `just check-llm`.

`check-llm` is non-mutating (no autofix). If it fails on lint/format, run `just lint-fix`,
then re-run the gate.

Never substitute a pipeline- or framework-generic gate set (e.g. bare `pytest` +
`ruff check` + `lint-imports` from an external tool's defaults) for this gate — when any
orchestration (skills, sub-agents, CI scaffolding) asks "what are the quality gates", the
answer is `just check-llm`, matching `quality_gates` in `milknado.toml`.

## Key Recipes

```bash
just install        # Install all dependencies (uv sync)
just lint           # Ruff check + format check (no changes — used by CI)
just lint-fix       # Ruff check + format with autofix
just test           # Run pytest (supports args: just test -k pattern)
just test-file <f>  # Run a single test file
just check-llm      # THE PR GATE — lint + format + tests + project & diff coverage, quiet on success
just build          # Full pipeline with autofix (lint-fix → test → coverage)
just build-ci       # Full pipeline no autofix — CI uses this
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
- **Tests**: `tests/` — pytest, 95% coverage required (matches `codecov.yml`)

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

## Code-Intelligence Routing

Three MCP servers cover code intelligence; they layer rather than overlap. The
worktree's `.mcp.json` wires serena and code-review-graph; tilth is provided
globally. Prefer them over raw `grep`/`cat`/`sed`/`Edit`.

- **tilth** — file I/O floor. Default for read/search/edit; replaces host
  Grep/Read/Edit/Glob. Always search first (`tilth_search`), then read
  (`tilth_read`), then edit.
- **serena** — LSP-grounded symbol layer. Use when ground-truth semantics
  matter (overloads, generics, dispatch, type info) or for symbol-bounded edits.
- **code-review-graph** — project-scale graph. Use for blast-radius,
  "what does this codebase do", and review-scope queries — not routine search.

### Editing: serena vs tilth

Pick by edit *shape*, not preference. Serena is more context-efficient for
symbol-bounded edits (no need to re-ship the surrounding body); tilth wins for
everything else and for the read-step that precedes either.

| Edit shape | Pick |
|---|---|
| Replace whole function / method / class body | `serena.replace_symbol_body` |
| Insert relative to a known symbol | `serena.insert_before_symbol` / `insert_after_symbol` |
| Rename a symbol across the codebase | `serena.rename_symbol` |
| Safe-delete an unused symbol | `serena.safe_delete_symbol` |
| Sub-symbol edit (slice inside a function) | `tilth_edit` hash-anchor |
| Imports, config (TOML/JSON/YAML), Markdown, shell | `tilth_edit` |
| Create new file | `tilth_edit` overwrite |
| Bulk pattern across files | `tilth_search` + `tilth_edit` batch |

Read-step: serena's `get_symbols_overview` + `find_symbol(include_body=true)`
pulls only the target symbol out of a large file — reach for that when you need
one function from a long file rather than reading the whole thing.

### Routing self-check

Before each tool call, ask the shape of the question:

- Symbol-level read or edit → **serena**
- File-level read, search, or edit → **tilth**
- Graph-level analysis (impact, architecture, flows) → **code-review-graph**

If unsure, pick the smallest-scope tool that answers the question.
