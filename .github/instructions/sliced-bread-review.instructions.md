---
applyTo: "src/milknado/**/*.py"
excludeAgent: "coding-agent"
---

## Sliced Bread deep review — slices, layers, and barrels

Layout: pure domain slices under `src/milknado/domains/` (batching, common, config_sync, dispatch, execution, github, graph, planning, reporting, wiki), application wiring in `app/`, delivery in `cli/` and `mcp/`, adapters implementing domain ports in `adapters/`. Each domain's `__init__.py` barrel is its public API. (Diffs to a barrel itself get the stricter contract review in `barrel.instructions.md`.)

import-linter (`lint-imports`, part of `just check-llm`) already enforces the mechanical contracts — domains never import app/cli/mcp, graph never imports execution, app never imports cli/mcp, adapters never import app, and boundary layers use domain barrels — so do not re-flag plain forbidden imports. Review for what the checker cannot see, in priority order:

### 1. Contract evasion

- A dynamically built import (`importlib.import_module`, string-assembled module paths) that dodges import-linter's static scan is the one mechanical breach it cannot see — flag it.
- A new underscore-private module inside a domain that is not added to the "boundary modules must use domain barrels" contract list in `pyproject.toml` silently becomes importable by app/cli/mcp/adapters — flag the missing contract entry.

### 2. Misplaced invariants

- **Caller-shadowed invariant**: a guard or validation every caller must run lives outside the producing slice. Signals: the same check repeated at N call sites around one producer; a helper exported from a barrel solely so `app/` or another slice can call it; some callers apply the check while others skip it. Fix: move the invariant into the producer so callers inherit it, then narrow the barrel.
- **Exported special case**: an error, empty/boundary case, or default that every caller handles identically when the producer has the information to absorb it.

### 3. Slice discipline

- Cross-slice translation lives in an explicit bridge module (e.g. `domains/planning/batching_bridge.py`), never inside the pure slice.
- `common/` is a leaf — flag anything in `common/` importing a sibling domain.
- Flag new single-use factories, registries, abstract bases, or shared utilities added without a second concrete consumer.

### Severity framing

Grade by boundary position: barrel/public surface > cross-slice > intra-slice > file-private. The same leak that is minor inside one file is a blocker on a barrel surface.
