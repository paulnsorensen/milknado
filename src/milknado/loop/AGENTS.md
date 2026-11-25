# Loop engine

`milknado.loop` is the worker runtime. Milknado owns this code. It is not a verbatim vendor copy.
These rules add to the root `AGENTS.md` for files under this directory.

## Public crust

- The public API is `__all__` in `__init__.py`. Keep it small. Add a name only for a consumer outside this package.
- Other packages import from `milknado.loop` only. Do not add an outside import of a private `_module` here.
- `adapters/` here holds the per-agent-CLI adapters (Claude, Codex, Copilot, Crush, OMP, OpenCode). It is not `milknado.adapters`.

## Imports

- Loop code may import shared values from `milknado.domains.common`.
- Do not import or name `milknado.adapters`, `milknado.app`, `milknado.cli`, or `milknado.mcp`, also not as a module string.
- `_process_helper.py` starts `milknado.adapters._loop_lifeline` by name. This hidden cycle is audited debt. Do not add another.

## Behavior to keep

- Keep turn counting, hard turn caps, Claude and Codex soft wind-down, and subprocess containment.
- Drain reader threads before you close process pipes, so that late output stays visible.
- Tests for this package are in `../../../tests/loop/`.

## Size (narrows root "Code Style")

- `_agent.py` and `engine.py` exceed 300 lines. Do not grow them. Extract a sibling module.
