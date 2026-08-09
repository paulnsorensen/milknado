# Entry Points — CLI & MCP Server

Milknado ships two executables, both declared in `pyproject.toml` `[project.scripts]`:

- `milknado` → `milknado.cli:app` — a Typer CLI for humans.
- `milknado-mcp` → `milknado.mcp.server:main` — a FastMCP **stdio** server for agent
  hosts (Cursor, Codex, Gemini CLI, Claude Code, etc.).

Both are thin facades. They parse arguments / requests, resolve config + open the graph DB,
then hand off to the domain slices under `src/milknado/domains/`. Neither holds Mikado logic
of its own — that lives in `planning`, `execution`, `batching`, `dispatch`, `graph`, `common`.

## Why two

The CLI is the interactive front door (plan a spec, run the loop, inspect state). The MCP
server exposes the same engine as tools so an agent in another harness can drive the graph
in-loop — adding nodes, polling detached runs, walking the todo tree — without shelling out.
They share the domain layer; they do not share a process or import each other.

## Boundary ownership

Project-root resolution and graph/config opening are shared application policy in
`src/milknado/app/project.py`; `milknado.project` supplies the presentation-free
`OpenProject` bootstrap. CLI helpers may render plugin and parent-block messages, but they
route project loading through that shared bootstrap rather than discovering policy separately.

Application policy modules are presentation-free: `app/plan.py` returns planning results and
application errors, and `app/run.py` returns typed protected-branch refusals. Rich rendering,
interactive prompts, and Typer exit-code mapping belong to `cli/plan.py` and `cli/run.py`.
MCP handlers use the parsers in `mcp/_core.py` for external enum normalization and return
structured values/errors without writing to stdout or stderr.

The `lint-imports` gate enforces these directions: domains cannot import app/cli/mcp; app
cannot import cli/mcp; adapters cannot import application policy; and boundary modules use
owning domain barrels instead of direct domain submodules. Adapter dependencies on domain
ports and types are allowed. Internal domain modules are not a public import surface, so a
new direct bypass must be removed or documented as a narrow architectural exception before
it is admitted.

## CLI surface (`milknado`)

`cli/` is a **package**; `cli/__init__.py` is the facade — it builds the `typer.Typer` app
(exported as `app`) and mounts the sub-apps + commands defined in sibling modules. Each
command group lives in its own module so no file blows the 300-line cap.

- **Top-level commands** (`cli/__init__.py`): `init` (create `milknado.toml`, init DB + CRG
  index, optionally install Rust tools, wire worker hooks), `index` (rebuild CRG), `status`
  (render the graph tree, enriched with live ralph run states), `rebalance` (sweep finished
  subtrees, regroup orphans, reap worktrees — see [[graph-status-reconciliation]]),
  `crg` (architecture overview), `add-node`, `doctor` (health checks).
- **Planning** (`cli/plan.py`): the `plan` command. Accepts `--spec` (.md) and/or `--issue`
  (GitHub refs fetched via `gh`), materializes a combined spec into `.milknado/issues/`,
  derives a goal from the first `# heading`, and delegates to `domains.planning.Planner`.
  Supports `--interactive` accept/revise/cancel iteration.
- **Execution** (`cli/run.py`): the `run` command (parallel ralph loops over ready leaf
  nodes) and `attach` (join an in-flight run's TUI).
- **Agents** (`cli/agents.py`): `agents check` — prints resolved planning/execution commands
  and a sample planning argv (with env/stdin redacted).
- **Config** (`cli/config.py`): `config show` — the effective, layered configuration
  (see [[config-layering]]).
- **Graph edits** (`cli/graph.py`): `graph archive` / `graph unarchive` (archive lifecycle)
  and `edge add` (direct edge operations).
- **Roadmaps** (`cli/roadmap.py`, `cli/github_roadmap.py`): `roadmap import|export` (to/from
  the hallouminate wiki) and `github-roadmap import|bind|export` (GitHub Projects v2).
- **Tools / plugins** (`cli/tools.py`): `tools check|install` (Rust toolchain status via
  `domains.common.toolchain`), `plugin init` (scaffold), plus the worker-hook writers that
  merge `rtk hook <family>` into per-harness settings (`.claude`, `.gemini`, cursor hooks).
- **Shared helpers** (`cli/_helpers.py`): `_find_config`, `_load_or_default`, `_open_project`,
  `_maybe_block_parent` — commands resolve config and open the project through `_open_project`
  (the shared `OpenProject` bootstrap), then `graph.close()` in a `finally`.

## MCP surface (`milknado-mcp`)

The shared `FastMCP` instance and the project-root/graph helpers (`resolve_project_root`,
`open_graph`, status/kind parsers, the unified `RunDict` return schema) live in
**`src/milknado/mcp/_core.py`** — split out so the tool modules register against one `mcp` without forming
an import cycle. Project root comes from the `project_root` arg or `MILKNADO_PROJECT_ROOT`.

`main()` in `mcp/server.py` imports the tool modules — `github`, `node`, `ralph`,
`rebalance`, `run`, `todo`, `todo_mutate`, `wiki` — purely for their **import side effect**:
each module's `@mcp.tool()` decorators register on the shared instance at import time. It then
calls `mcp.run()`. (`mcp/server.py` itself also defines `milknado_graph_summary`,
`milknado_plan_batches`, and `milknado_plan_apply`.) Tools are grouped by capability:

- **Todo workflow — read** (`mcp/todo.py`): `milknado_todo_tree`, `milknado_todo_next`,
  `milknado_todo_brief`, `milknado_get_node` — brief rendering delegates to
  `domains.dispatch.render_brief`.
- **Todo workflow — mutate** (`mcp/todo_mutate.py`): `milknado_todo_add`,
  `milknado_todo_set_status`, `milknado_set_subtree_status`, `milknado_archive_node`,
  `milknado_unarchive_node`, `milknado_track_follow_up`, `milknado_delete_node`,
  `milknado_move_node`, `milknado_edit_node` — node lifecycle enforced through
  `VALID_TRANSITIONS`.
- **Node claim / verify** (`mcp/node.py`): `milknado_goal_claim`, `milknado_goal_release`,
  `milknado_todo_claim`, `milknado_node_verify`.
- **Planning** (`mcp/server.py`): `milknado_plan_batches` accepts file changes + new
  relationships (hash anchors / symbol refs) and turns them into a batch plan, guarding
  against mega-batches over a change threshold; `milknado_plan_apply` writes a caller-produced
  `milknado.plan.v2` manifest to the graph as nodes.
- **Run start / poll / cancel** (`mcp/run.py`, `mcp/ralph.py`): `milknado_run_inline` (sync),
  `milknado_run_inline_start` / `milknado_run_inline_poll`, `milknado_run_loop_start` /
  `milknado_run_loop_poll` (detached worktree + full ralph loop), `milknado_run_list`,
  `milknado_run_cancel`, plus `milknado_deposit_result` / `milknado_deposit_review` for worker
  hand-back. Detached runs claim the node RUNNING via a SQLite conditional UPDATE (cross-process
  mutual exclusion) and survive a server restart; pollers reconcile orphaned or dead-runner
  runs before reporting (see [[run-owner-recovery]]).
- **Rebalance** (`mcp/rebalance.py`): `milknado_rebalance` — the MCP surface of the `rebalance`
  CLI command.
- **Roadmaps** (`mcp/wiki.py`, `mcp/github.py`): `milknado_roadmap_import` /
  `milknado_roadmap_export` (hallouminate wiki) and `milknado_github_roadmap_import` /
  `milknado_github_roadmap_bind` / `milknado_github_roadmap_export` (GitHub Projects v2).

### Python-version schema compatibility

FastMCP evaluates MCP tool return annotations through Pydantic when decorators register the tools. Because Milknado supports Python 3.11, `RunDict` must inherit `typing_extensions.TypedDict`; `typing.TypedDict` cannot supply the runtime metadata Pydantic needs on Python 3.12 and lower.[^typed-dict-contract] Keep `typing-extensions` as a direct project dependency rather than relying on Pydantic's transitive installation.[^typed-dict-dependency]

[^typed-dict-contract]: `src/milknado/mcp/_core.py:11-14,40`; PR #325 Actions job 89758215559
[^typed-dict-dependency]: `pyproject.toml:9,21-22`; PR #325 commit `5889c635a07627461d6acab44a92adb6ae90ff17`

_Source: PR #325 affinage · Updated: 2026-07-25 · Supersedes: none_

## stdio gotcha

The MCP server speaks **stdio**. Restarting it (e.g. `just mcp-dev` watches `src/milknado/`
and restarts on every change) drops the existing stdio connection — the client must `/mcp`
reconnect to pick up the new process and the freshly-registered tools.

## See also

- `pyproject.toml` `[project.scripts]` — the two console-script entry points.
- `src/milknado/domains/` — where both facades delegate the actual engine logic.
