# Milknado

See [AGENTS.md](AGENTS.md) for full development instructions.

**Run `just build` before opening any PR.**

## Code-Intelligence Routing

Three MCP servers cover code intelligence; they layer rather than overlap. The
worktree ships an `.mcp.json` wiring all of them — prefer them over raw
`grep`/`cat`/`sed`/`Edit`.

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
