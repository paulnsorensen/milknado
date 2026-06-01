# Planning Domain — Goal Decomposition & Manifest

The planning slice (`src/milknado/domains/planning/`) turns a natural-language goal
into a Mikado dependency graph. It drives a planning agent (subprocess), parses its
output into a validated manifest, batches the changes, and writes the batches into the
graph as nodes. It is the front half of the engine; the batching slice is the back half.

## Data flow (the launch pipeline)

`Planner.launch(goal, project_root, spec_path=?)` in `planner.py` runs these steps:

1. **Build context** — `build_planning_context` (`context.py`) assembles a markdown
   prompt: goal, compact CRG architecture overview, tilth structural map, the existing
   Mikado graph state, a batching-policy note, the v2 manifest schema + instructions, and
   (optionally) the spec text. Written to `<root>/.milknado/planning-context.md`.
2. **Run the agent** — `build_planning_subprocess` (in `domains/common/agent_argv`)
   builds the argv; the agent runs as a subprocess with stdout piped.
3. **Parse manifest** — `parse_manifest_from_output` (`manifest.py`) extracts a single
   fenced ```json block from stdout and validates it. No manifest → `PlanResult` with
   `solver_status="NO_MANIFEST"`.
4. **Optional validation hook** — an external command (`planning_validation_hook`) is fed
   the manifest as JSON on stdin; non-zero exit rejects the plan (`VALIDATION_FAILED`).
5. **Batch** — `run_batching` (`batching_bridge.py`) calls into the batching slice.
6. **Apply to graph** — `apply_batches_to_graph` writes nodes/edges.
7. **Telemetry** — `record_batch_snapshot` appends a JSONL calibration record.

`replan_with_delta` is just `launch` with a delta goal — resume is handled by the context
builder detecting existing nodes, not by a separate code path.

## The manifest (`manifest.py`)

`PlanChangeManifest` is the validated contract between agent and engine. Version string
`milknado.plan.v2` (`MANIFEST_VERSION`) is checked exactly — a mismatch rejects the
manifest. Fields: `goal`, `goal_summary` (becomes the root node description; every
executor reads it), optional `spec_path`, a tuple of `FileChange`, and a tuple of
`NewRelationship`.

Parsing is **fail-loud at the boundary** (Principle 1/2): every field is type-checked and
on any violation the parser logs a warning and returns `None` rather than raising. Key
invariants enforced during parse:

- `edit_kind` ∈ {add, modify, delete, rename}; `new_relationships[].reason` ∈
  {new_file, new_import, new_call, new_type_use} — closed enums.
- Change `id`s are unique; every `depends_on` id and every relationship endpoint must
  reference a known change id.
- `goal`, `goal_summary`, and each change `description` must be non-empty.
- `hash_anchors` (before/after edit-span hashes) and per-change `dependencies` are
  optional but, if present, fully validated.

`FileChange`, `SymbolRef`, `HashAnchors`, `NewRelationship` etc. are imported from the
batching slice (`change.py`) — the manifest reuses the batching domain's types rather than
defining its own. `manifest_to_dict` serializes back for the validation hook payload.

## Context assembly (`context.py`)

`build_planning_context` joins independent section builders with blank lines. Notable
shape decisions:

- **Degradation, not failure**: if CRG or tilth is unavailable, the section prints a
  `_(... skipped)_` marker and planning continues. `_safe_ensure_crg` in `planner.py`
  catches CRG init failures and proceeds without structural edges.
- **Resume awareness**: `resuming = len(graph.get_all_nodes()) > 0`. The instructions
  section switches to "do NOT recreate existing nodes" and the graph section dumps
  progress counts, failed nodes (need re-planning), and ready-to-execute nodes.
- **Schema is embedded in the prompt** — `_instructions_section` ships the full v2 JSON
  schema, enum constraints, granularity guidance ("emit file-level changes, let the solver
  batch"), and the MCP-targeting note (require tilth/CRG inspection for real hash_anchors).
- Guard: `spec_text == ""` raises — callers must pass `None` to omit the spec, never empty.

## Bridge into batching (`batching_bridge.py`)

`run_batching` calls `plan_batches` with `budget=DUMB_ZONE_BUDGET` (120K) and a 10s time
limit. `apply_batches_to_graph` then materializes the `BatchPlan`:

- If `parent_id is None` (fresh graph), it creates a goal-root node from `goal_summary`
  and attaches root batches to it; raises `ValueError` if goal/goal_summary are empty.
  If a graph already has a root, batches attach under the existing parent — no new root.
- Each batch becomes one node (carrying `oversized` and `batch_index`). Inter-batch
  `depends_on` become graph edges; the lexicographically-smallest dep is the primary
  parent. Batches with no deps attach to the root/parent. File ownership is set per node.
- Empty manifest returns `[goal_root_id]` (fresh) or `[]` (existing parent).

`PlanResult` summarizes the run: `nodes_created`, `batch_count`, `oversized_count`,
`solver_status`, `change_count`, plus `success`/`exit_code`.

## Telemetry (`telemetry.py`)

`record_batch_snapshot` appends one JSONL line to `.milknado/calibration.jsonl` per plan:
change/batch/oversized counts, solver status, symbol-spread max/mean, relationship count.
Uses a single atomic `os.write()` on an `O_APPEND` fd so concurrent appends from parallel
processes never interleave. **Never raises** — telemetry must not block the planner; OS
errors are swallowed with a warning.

## Gotcha: mega-batch guard

`_check_mega_batch` exists in `planner.py` (raises `MegaBatchAborted` when a single batch
exceeds a change threshold without `force_single_batch`) but is not wired into `launch`.
`<speculative>` it is invoked by a higher-level caller (CLI/MCP) rather than the Planner
itself.
