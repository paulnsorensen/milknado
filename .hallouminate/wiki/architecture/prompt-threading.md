# Prompt Threading — how briefs and worker definitions reach a dispatch

One resolver, two prompt pipelines, three prepend knobs. The map that keeps
them straight.

## Kind gates *whether*, flavor picks *which*

Only `TASK` nodes are ever dispatched — `milknado_todo_claim`
(`mcp_node.py:189-192`) and `_claim_ralph` (`mcp_ralph.py:88-91`) both raise on
non-TASK. `GOAL` is claimed only as a coordinator fence (`goal_claims` row);
`ROADMAP` is a pure container. `flavor` exists only on TASK nodes
(`types.py:86-95`).

## One resolver: `resolve_flavor_profile`

`resolve_flavor_profile(cfg, flavor)` (`domains/common/flavor_profile.py:35-124`)
merges cfg-level defaults with the `[milknado.flavor.<name>]` override per field:

- `execution_agent` — explicit override wins; else **derived from `tools`** via
  `resolve_worker_tools` → `resolve_execution_agent_command`; else cfg's.
- `quality_gates` / `brief_prepend` / `agent_type` / `loop_mode` /
  `max_iterations` / `max_turns` — override if not `None`, else cfg.
- `worktree` — override if not `None`, else `True` (no cfg-level default exists).

`agent_type` is **opaque pass-through**: type-checked as a string only, resolved
by the harness (Claude Code `Workflow`/`agent()`), never validated against any
agent registry — contrast `execution_agent`, whose executable basename must be
one of `{claude, codex, cursor-agent, gemini}` (`agent_argv.py`). Default
`agent_type` is `milknado:milknado-worker`, defined at
`plugins/milknado/agents/milknado-worker.md` (Edit/Write/Read/Bash +
`milknado_node_verify`/`milknado_deposit_result`/`milknado_track_follow_up` —
no coordinator tools).


### Path provenance

The two file-backed prompt loaders share the dependency-neutral `resolve_project_path` helper. Project-local prompt and flavor prepend paths are confined to `project_root`, including absolute paths and symlink-resolved targets. Global paths are resolved against the global config directory before the layered merge and carry trusted provenance, so a deliberate global path may live outside the project root.

The resolved flavor profile also carries `worktree`: an explicit flavor value wins, otherwise `[milknado] worktree` supplies the default (`true`).

## One brief pipeline

`render_brief` is the single node-context renderer for native/MCP and subprocess
ralph workers. Assembly order is `{brief_prepend}` → task/review/plate heading →
goal context → completed prerequisites → relevant files → nearest spec →
flavor-specific instructions.

The ralph adapter receives that rendered markdown and appends only ralph-loop
scaffolding: optional prior review findings, quality gates, follow-up protocol,
and the completion sentinel. `RALPH.md` remains a plain markdown prompt body,
which is the format the loop runner parses.

`ExecutionConfig.brief_prepend` carries the resolved profile through both
subprocess entry points: `RunLoop._dispatch_batch` and the detached
`_ralph_node_runner`. The base execution config starts from
`worker_brief_prepend`; a flavor profile replaces it per node.

## Three prepend knobs — don't conflate

| Knob | Feeds | Path |
|---|---|---|
| `[milknado.flavor.<name>] brief_prepend` (or `_path`) | worker brief header | native/MCP and subprocess ralph |
| `planning_prompt_prepend` (cfg) | goal-decomposition prompt (`cli_plan.py` via `planning/context.py:26-27`) | planning only |
| flavor `## Instructions` block | brief tail | static, code-owned (`brief.py`), only `review`/`plate` customized |

## Worker tool surfaces (who may call what)

Subprocess-spawned CLI workers get only `milknado_track_follow_up` +
`milknado_deposit_result` (`WORKER_ALLOWED_TOOLS`, `agent_argv.py`); the native
plugin agent adds `milknado_node_verify`. Run-dispatch tools are
coordinator-only — ralph carries a permanent prohibition against granting them
to workers (`mcp_ralph.py:1-12`). See
[Execution & Dispatch](./execution.md) for the single-shot vs ralph family table.
