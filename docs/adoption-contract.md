# easy-cheese adoption contract

This is the interface milknado promises to the easy-cheese ecosystem (skills,
`/ultracook`, `/cheese-factory`, `/wheypoint`, `/age`/`/cure`). It is
documentation, not code — the vocabulary and preset pack below ship from
easy-cheese, not from milknado (ADR-004). milknado stays workflow-agnostic:
it supplies the flavor-registry mechanism, the graph, and the MCP surface;
easy-cheese supplies the phase names and the skill-side wiring.

## Adoption is presence-gated

When milknado is present in a repo (`.milknado/` initialized, `milknado-mcp`
reachable), the Mikado graph is the **sole** session task surface — skills and
the orchestrator stop writing Claude Code's native Tasks list. When milknado
is absent, today's fallbacks (native Tasks, flat `.cheese/<phase>/<slug>.md`
files) are unchanged. There is no mirroring and no coexistence (ADR-001):
native Tasks cannot hold `flavor`/`artifact_path`, and a one-way mirror would
be lossy and add a drift surface. A skill checking adoption should look for
`.milknado/` at the repo root before routing task-tracking calls to milknado;
absent that, do not call any `milknado_*` tool.

### Named interim visibility gap (ADR-001)

No harness UI renders the Mikado graph today — the tmux status panel is a
deferred sibling goal, not part of this contract. Until it lands, a session
has no live view of its own task graph beyond what it queries. Mitigation:
call `milknado_todo_tree` (or `milknado_graph_summary`) at session
boundaries — start, after a batch of claims, before handoff — and render the
result inline. This is a known gap, not an oversight; do not build a
throwaway visualizer to paper over it.

## Native-Tasks → milknado mapping

| Native Claude Code Tasks | milknado equivalent | Notes |
|---|---|---|
| `TaskCreate` | `milknado_todo_add(description, kind="task", parent_id, flavor, artifact, prereqs)` | `kind` also covers `roadmap`/`goal` for the containment levels native Tasks has no notion of. |
| `blockedBy` | `prereqs=[node_id, ...]` on `todo_add`/`track_follow_up` | Cycle-checked at write time; rejected outright rather than silently dropped. |
| `blocks` (reverse) | not a separate call — expressed by the other task's `prereqs` | milknado edges are directional (`parent_id` prereq → `child_id` dependent); there is no reverse-declare tool, mirroring native Tasks' single-direction API. |
| `TaskUpdate(status=...)` | `milknado_todo_set_status(node_id, status)` / `milknado_set_subtree_status(root_id, status)` | `done` is server-gated: a natively-claimed node must have a passing `milknado_node_verify` on record first (fail-closed). |
| `TaskUpdate(content=...)` | `milknado_edit_node(node_id, description=...)` | Coordinator-only. |
| *(no equivalent)* | `artifact` param / `milknado_edit_node(..., artifact=...)` | Opaque repo-relative markdown reference to the node's narrative (spec, report, brief) — native Tasks has no deliverable field (ADR-002). |
| *(no equivalent)* | `flavor` param, validated against the flavor registry | Native Tasks is type-less; milknado's flavor drives model/tools/loop_mode/worktree policy per node. |
| `TaskList` scoped to a session | `milknado_goal_claim(goal_id, owner)` / `milknado_goal_release` | A session claims a GOAL node's subtree as its task list (ADR-003); dead-pid claims are reclaimable. |
| *(no equivalent — flat list)* | `milknado_todo_tree` / `milknado_graph_summary` | Graph-shaped view instead of a flat list; use at session boundaries per the visibility-gap mitigation above. |
| *(no equivalent)* | `milknado_todo_claim(node_id, worktree=None)` | Claims a task RUNNING and provisions (or skips) a worktree per flavor policy (ADR-005) — native Tasks has no execution-isolation concept. |
| *(no equivalent)* | `milknado_node_verify(run_id)` | Flavor-appropriate completion evidence (ADR-006): stageable diff for worktree nodes, artifact-existence for in-place nodes. Required before a natively-claimed node can go `done`. |

## Phase-flavor preset pack

The phase vocabulary below is an easy-cheese opinion, not a milknado
built-in (ADR-004). Install it globally (`~/.config/milknado/milknado.toml`)
so every repo with milknado present resolves these flavors without a
per-repo copy; a repo-local `milknado.toml` may still override individual
knobs per the existing defaults → global → local precedence.

`brainstorm`/`explore`/`agent-review`/`harden`/`fix` map to `/mold`,
`/culture`+explorer-shaped work, `/age`, `/press`, and `/cure` respectively.
`spec`/`implement`/`research` are milknado's built-in flavor names,
re-profiled here to bind them to easy-cheese's phase agents via
`agent_type` instead of the milknado default worker.

```toml
# ~/.config/milknado/milknado.toml

[milknado.flavor.brainstorm]      # /mold — fuzzy idea -> spec, no code changes
agent_type = "generalist"
worktree = false
loop_mode = "single"
quality_gates = []

[milknado.flavor.explore]         # /culture, or any read-only investigation
agent_type = "explorer"
worktree = false
loop_mode = "single"
quality_gates = []

[milknado.flavor.spec]            # built-in flavor, re-profiled: written narrative, no code
agent_type = "Plan"
worktree = false
loop_mode = "single"
quality_gates = []

[milknado.flavor.implement]       # built-in flavor, re-profiled: TDD code change in isolation
agent_type = "coder"
worktree = true
loop_mode = "redispatch"
quality_gates = ["just check-llm"]

[milknado.flavor.research]        # built-in flavor, re-profiled: external-fact lookups
agent_type = "researcher"
worktree = false
loop_mode = "single"
quality_gates = []

[milknado.flavor.agent-review]    # /age — read-only, severity-tagged findings
agent_type = "reviewer"
worktree = false
loop_mode = "single"
quality_gates = []

[milknado.flavor.harden]          # /press — add hardening tests on top of a cook
agent_type = "roquefort-wrecker"
worktree = true
loop_mode = "redispatch"
quality_gates = ["just check-llm"]

[milknado.flavor.fix]             # /cure — apply findings, then gate
agent_type = "coder"
worktree = true
loop_mode = "redispatch"
quality_gates = ["just check-llm"]
```

`brainstorm`/`explore`/`agent-review` are `artifact`-bearing, gate-empty,
in-place (`worktree = false`) nodes: their deliverable is prose, checked at
`milknado_node_verify` time via artifact-existence (ADR-006), not a diff.
`implement`/`harden`/`fix` are worktree-isolated with `just check-llm` as
their gate, keeping the existing stageable-change completion check.

## Ultracook worked example

`/ultracook <spec>` pipelines a spec through `cook → press → age → cure`
phases. Under milknado, each curd (independent implementation slice) is a
worktree-isolated `implement` node; the phases that read/write across curds
(wiring, review, cure) are in-place nodes with `prereqs` on every curd they
depend on.

```
GOAL "ultracook: <spec-slug>"                       (goal_claim'd by the session)
├── TASK "curd: auth-slice"        flavor=implement  worktree=true
├── TASK "curd: billing-slice"     flavor=implement  worktree=true
├── TASK "curd: notif-slice"       flavor=implement  worktree=true
├── TASK "wire curds together"     flavor=fix         worktree=false  prereqs=[auth, billing, notif]
├── TASK "press: harden tests"     flavor=harden      worktree=true   prereqs=[wire]
├── TASK "age: review"             flavor=agent-review worktree=false prereqs=[press]
└── TASK "cure: apply findings"    flavor=fix          worktree=true  prereqs=[age]
```

Sketch of the calls that build this graph (ids illustrative):

```
goal = milknado_todo_add("ultracook: <spec-slug>", kind="goal")
milknado_goal_claim(goal.id, owner=os.environ["SESSION_ID"])

auth     = milknado_todo_add("curd: auth-slice",    kind="task", parent_id=goal.id, flavor="implement")
billing  = milknado_todo_add("curd: billing-slice", kind="task", parent_id=goal.id, flavor="implement")
notif    = milknado_todo_add("curd: notif-slice",   kind="task", parent_id=goal.id, flavor="implement")

wire = milknado_todo_add(
    "wire curds together", kind="task", parent_id=goal.id, flavor="fix",
    prereqs=[auth.id, billing.id, notif.id],
)
press = milknado_todo_add("press: harden tests", kind="task", parent_id=goal.id,
                           flavor="harden", prereqs=[wire.id])
age   = milknado_todo_add("age: review", kind="task", parent_id=goal.id,
                           flavor="agent-review", artifact=".cheese/age/<slug>.md",
                           prereqs=[press.id])
cure  = milknado_todo_add("cure: apply findings", kind="task", parent_id=goal.id,
                           flavor="fix", artifact=".cheese/cure/<slug>.md",
                           prereqs=[age.id])
age_run = milknado_todo_claim(age.id, worktree=False)
# Dispatch the age worker; it writes the declared .cheese/age/<slug>.md artifact.
milknado_node_verify(age_run["run_id"])
```

Each curd runs in its own worktree and merges independently; the wiring node
runs in the current checkout (`worktree=false`) because it needs to see all
three curds' merged results at once — worktree isolation would hide them
from each other. `age` declares its findings report as its own `artifact`, so
verification requires that report to exist before `cure` becomes runnable.
`cure` reads the age report per the ingestion convention below and records its
own deliverable at `.cheese/cure/<slug>.md`.

## wheypoint + severity-gated findings ingestion

- **`/wheypoint`** checkpoints: when a session hands off, its `wheypoint`
  document path becomes the `artifact` on the GOAL (or the in-progress TASK)
  it summarizes, via `milknado_edit_node(node_id, artifact=".cheese/notes/<slug>.md")`.
  A resuming session reads that artifact before calling `milknado_todo_next`
  — the graph gives *what's next*, the artifact gives *why*.
- **Findings ingestion**: `/age` and `/affinage` write a severity-grouped
  report (blocker/high/medium/low). Only `blocker`/`high` findings become
  graph nodes — each surviving finding is one `milknado_track_follow_up`
  call (`flavor="fix"`, `artifact` pointing at the finding's slice of the
  report, `parent_id` under the cure GOAL). `medium`/`low` findings stay in
  the report only; they are not tracked as nodes unless a human promotes
  them. This mirrors the existing ADR: per-finding leaves severity-gated
  under a `cure` GOAL, not a flat dump of every finding into the graph.
