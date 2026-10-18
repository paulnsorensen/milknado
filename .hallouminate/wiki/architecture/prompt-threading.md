# Prompt Threading — how briefs and worker definitions reach a dispatch

One resolver, two prompt pipelines, three prepend knobs. The map that keeps
them straight.

## Kind gates *whether*, flavor picks *which*

Only `TASK` nodes are ever dispatched — `milknado_todo_claim`
(`mcp/node.py:135-136`) and `_claim_loop` (`app/loop.py:71`) both raise on
non-TASK. `GOAL` is claimed only as a coordinator fence (`goal_claims` row);
`ROADMAP` is a pure container. `flavor` exists only on TASK nodes
(`types.py:86-95`).

## One resolver: `resolve_flavor_profile`

`resolve_flavor_profile(cfg, flavor)` (`domains/common/flavor_profile.py:35-124`)
merges cfg-level defaults with the `[milknado.flavor.<name>]` override per field:

- `execution_agent` — explicit override wins; else **derived from `tools`** via
  `resolve_worker_tools` → `resolve_execution_agent_command`; else cfg's.
- `quality_gates` / `brief_prepend` / `max_iterations` — override if not
  `None`, else cfg.
- `worktree` — override if not `None`, else `True` (no cfg-level default exists).

`execution_agent`'s executable basename must be one of
`{claude, codex, cursor-agent, gemini}` (`agent_argv.py`). The former
`agent_type`, `loop_mode`, and `max_turns` flavor fields fed the in-session
Workflow path and were removed with it (2026-10-17; see
`history/workflow-executor-decision.md` §Removal).


### Path provenance

The two file-backed prompt loaders share the dependency-neutral `resolve_project_path` helper. Project-local prompt and flavor prepend paths are confined to `project_root`, including absolute paths and symlink-resolved targets. Global paths are resolved against the global config directory before the layered merge and carry trusted provenance, so a deliberate global path may live outside the project root.

The resolved flavor profile also carries `worktree`: an explicit flavor value wins, otherwise `[milknado] worktree` supplies the default (`true`).

## One brief pipeline

`render_brief` is the single node-context renderer for native/MCP and subprocess
loop workers. Assembly order is `{brief_prepend}` → task/review/plate heading →
goal context → completed prerequisites → relevant files → nearest spec →
flavor-specific instructions.

The loop adapter receives that rendered markdown and appends only loop
scaffolding: optional prior review findings, quality gates, follow-up protocol,
and the completion sentinel. `LOOP.md` remains a plain markdown prompt body,
which is the format the loop runner parses.

`ExecutionConfig.brief_prepend` carries the resolved profile through both
subprocess entry points: `RunLoop._dispatch_batch` and the detached
`_loop_node_runner`. The base execution config starts from
`worker_brief_prepend`; a flavor profile replaces it per node.

## Three prepend knobs — don't conflate

| Knob | Feeds | Path |
|---|---|---|
| `[milknado.flavor.<name>] brief_prepend` (or `_path`) | worker brief header | native/MCP and subprocess loop |
| `planning_prompt_prepend` (cfg) | goal-decomposition prompt (`cli/plan.py` via `planning/context.py`) | planning only |
| flavor `## Instructions` block | brief tail | static, code-owned (`brief.py`), only `review`/`plate` customized |

## Worker tool surfaces (who may call what)

Subprocess-spawned CLI workers get only `milknado_track_follow_up` +
`milknado_deposit_result` (`WORKER_ALLOWED_TOOLS`, `agent_argv.py`). A harness
session that claims a node with `milknado_todo_claim` verifies it with
`milknado_node_verify`. Run-dispatch tools are
coordinator-only — loop carries a permanent prohibition against granting them
to workers (`mcp/loop.py:1-12`). See
[Execution & Dispatch](./execution.md) for the single-shot vs loop family table.


_Source: branch refactor/remove-node-runner (edd6280) · Updated: 2026-10-17 · Supersedes: agent_type, loop_mode, and max_turns flavor fields_

