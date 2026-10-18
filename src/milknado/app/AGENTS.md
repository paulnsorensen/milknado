# Application layer

These rules add to the root `AGENTS.md` for files under this directory.
Source: `../../../docs/adr/sliced-bread-crust-003.md`.

## Ownership

- `app` owns application policy and composition. It opens the project, builds adapters, and calls domain barrels.
- `../cli/` and `../mcp/` call `app` functions. When they need new policy, add it here once.
- Give each function that CLI or MCP calls a public name. Delivery code must not import a `_private` app name.

## Presentation

- Policy modules (`run.py`, `plan.py`, `node.py`, `loop.py`, `project.py`) return typed results and typed errors. They do not print, prompt, or map exit codes. The CLI does that.
- Rich and Textual stay in the TUI modules: `*_tui.py`, `*_view.py`, `*_panels.py`, `*_overlays.py`, `*_commands.py`, `*_navigation.py`, `graph_display.py`, `run_view_app.py`, and `session_changes.py`.
- `run.py` must not import `run_tui.py`. `lint-imports` enforces this.

## Imports

- Do not import `milknado.cli`, `milknado.mcp`, or `milknado.web`. `lint-imports` enforces this.
- Do not name a `milknado.cli` or `milknado.mcp` module in a string either, for example a `python -m` target. Import-linter cannot see string references.
- Import adapters from the `milknado.adapters` barrel, not from adapter submodules.

## Size (narrows root "Code Style")

- `run.py` exceeds 300 lines. Do not grow it. Extract a sibling module.
