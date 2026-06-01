# Entry Points — CLI & MCP Server

Milknado ships two executables, both declared in `pyproject.toml` `[project.scripts]`:

- `milknado` → `milknado.cli:app` — a Typer CLI for humans.
- `milknado-mcp` → `milknado.mcp_server:main` — a FastMCP **stdio** server for agent
  hosts (Cursor, Codex, Gemini CLI, Claude Code, etc.).

Both are thin facades. They parse arguments / requests, resolve config + open the graph DB,
then hand off to the domain slices under `src/milknado/domains/`. Neither holds Mikado logic
of its own — that lives in `planning`, `execution`, `batching`, `dispatch`, `graph`, `common`.

## Why two

The CLI is the interactive front door (plan a spec, run the loop, inspect state). The MCP
server exposes the same engine as tools so an agent in another harness can drive the graph
in-loop — adding nodes, polling detached runs, walking the todo tree — without shelling out.
They share the domain layer; they do not share a process or import each other.

## CLI surface (`milknado`)

`cli.py` is the facade: it builds the `typer.Typer` app and mounts sub-apps + commands. The
big commands live in their own modules so no file blows the 300-line cap; `cli.py` only
re-exports and registers them.

- **Top-level commands** (`cli.py`): `init` (create `milknado.toml`, init DB + CRG index,
  optionally install Rust tools, wire worker hooks), `index` (rebuild CRG), `status` (render
  the graph tree, enriched with live ralph run states), `crg` (architecture overview),
  `add-node`, `doctor` (health checks).
- **Planning** (`cli_plan.py`): the `plan` command. Accepts `--spec` (.md) and/or `--issue`
  (GitHub refs fetched via `gh`), materializes a combined spec into `.milknado/issues/`,
  derives a goal from the first `# heading`, and delegates to `domains.planning.Planner`.
  Supports `--interactive` accept/revise/cancel iteration.
- **Execution** (`cli_run.py`): the `run` command. Builds an `ExecutionConfig`, wires
  `Executor` + `RunLoop` from `domains.execution`, and runs ready leaf nodes as parallel
  ralph loops on the current branch.
- **Agents** (`cli_agents.py`): `agents check` — prints resolved planning/execution commands
  and a sample planning argv (with env/stdin redacted).
- **Tools / plugins** (`cli_tools.py`): `tools check|install` (Rust toolchain status via
  `domains.common.toolchain`), `plugin init` (scaffold), plus the worker-hook writers that
  merge `rtk hook <family>` into per-harness settings (`.claude`, `.gemini`, cursor hooks).
- **Shared helpers** (`_cli_helpers.py`): `_find_config`, `_load_or_default`,
  `_ensure_plugins_loaded`, `_ensure_db` — every command resolves config, loads plugins, and
  opens a `MikadoGraph` through these, then `graph.close()` in a `finally`.

## MCP surface (`milknado-mcp`)

The shared `FastMCP` instance and the project-root/graph helpers (`resolve_project_root`,
`open_graph`, status/kind parsers, the unified `RunDict` return schema) live in
**`_mcp_core.py`** — split out so the tool modules register against one `mcp` without forming
an import cycle. Project root comes from the `project_root` arg or `MILKNADO_PROJECT_ROOT`.

`main()` in `mcp_server.py` imports `mcp_ralph`, `mcp_run`, `mcp_todo` purely for their
**import side effect** — each module's `@mcp.tool()` decorators register on the shared
instance at import time — then calls `mcp.run()`. Tools are grouped by capability:

- **Graph editing / inspection** (`mcp_server.py` + `mcp_todo.py`): `milknado_graph_summary`,
  `milknado_add_node`, `milknado_get_node`, `milknado_edit_node`, `milknado_delete_node`,
  `milknado_move_node`, `milknado_set_subtree_status`.
- **Planning** (`mcp_server.py`): `milknado_plan_batches` — accepts file changes + new
  relationships (with hash anchors / symbol refs) and turns them into a batch plan; guards
  against mega-batches over a change threshold.
- **Todo workflow** (`mcp_todo.py`): `milknado_todo_add`, `milknado_todo_brief`,
  `milknado_todo_next`, `milknado_todo_tree`, `milknado_todo_set_status`,
  `milknado_track_follow_up` — node lifecycle enforced through `VALID_TRANSITIONS`; brief
  rendering delegates to `domains.dispatch.render_brief`.
- **Run start / poll / cancel** (`mcp_run.py`, `mcp_ralph.py`): `milknado_todo_run` (sync),
  `milknado_todo_run_start` / `milknado_todo_run_poll`, `milknado_ralph_run_start` /
  `milknado_ralph_run_poll` (detached worktree + full ralph loop), `milknado_run_list`,
  `milknado_run_cancel`. Detached runs claim the node RUNNING via a SQLite conditional UPDATE
  (cross-process mutual exclusion) and survive a server restart; pollers reconcile orphaned
  or dead-runner runs before reporting.

## stdio gotcha

The MCP server speaks **stdio**. Restarting it (e.g. `just mcp-dev` watches `src/milknado/`
and restarts on every change) drops the existing stdio connection — the client must `/mcp`
reconnect to pick up the new process and the freshly-registered tools.

## See also

- `pyproject.toml` `[project.scripts]` — the two console-script entry points.
- `src/milknado/domains/` — where both facades delegate the actual engine logic.
