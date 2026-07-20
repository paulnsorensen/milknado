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

## Two prompt pipelines (they do not share a brief-builder)

**Path A — native/MCP** (`milknado_todo_claim`, `milknado_todo_brief`,
`milknado_run_inline*`, `dispatch_node_sync`): `render_brief`
(`domains/dispatch/brief.py:107-140`). Assembly order:
`{brief_prepend}` → `# Task: {description}` → `## Goal context` (ancestors) →
`## Prerequisites already done` → `## Relevant files` → `## Spec` (nearest
`artifact_path`) → `## Instructions` (static per flavor — only `review`/`plate`
have their own block; all others get the default coder instructions,
`brief.py:117-122`).

**Path B — subprocess ralph** (`milknado run` CLI, `RunLoop`,
`milknado_run_loop_start`): `_build_ralph_content` (`adapters/loop.py:254-288`)
writes `RALPH.md`; signature is `(node, context, quality_gates)`.
**`brief_prepend` is resolved and then silently dropped** — no prepend parameter
exists on this path (`run_loop/__init__.py:308-313`,
`_ralph_node_runner.py:79-85`). Per-flavor prompt text set in TOML reaches
Path A workers verbatim and Path B workers not at all. Known gap, tracked in
the issue tracker; `FlavorProfile`'s docstring documents `worker_agent_type`/
`loop_mode`/`max_*` as native-only but says nothing about `brief_prepend`.

## Three prepend knobs — don't conflate

| Knob | Feeds | Path |
|---|---|---|
| `[milknado.flavor.<name>] brief_prepend` (or `_path`) | worker brief header | Path A only (see gap above) |
| `planning_prompt_prepend` (cfg) | goal-decomposition prompt (`cli_plan.py` via `planning/context.py:26-27`) | planning only |
| flavor `## Instructions` block | brief tail | static, code-owned (`brief.py`), only `review`/`plate` customized |

## Worker tool surfaces (who may call what)

Subprocess-spawned CLI workers get only `milknado_track_follow_up` +
`milknado_deposit_result` (`WORKER_ALLOWED_TOOLS`, `agent_argv.py`); the native
plugin agent adds `milknado_node_verify`. Run-dispatch tools are
coordinator-only — ralph carries a permanent prohibition against granting them
to workers (`mcp_ralph.py:1-12`). See
[Execution & Dispatch](./execution.md) for the single-shot vs ralph family table.
