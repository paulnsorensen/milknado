# Planning Domain — Goal Decomposition & Manifest

The planning slice (`src/milknado/domains/planning/`) turns a natural-language goal
into a Mikado dependency graph. It drives a planning agent (subprocess), parses its
output into a validated manifest, batches the changes, and writes the batches into the
graph as nodes. It is the front half of the engine; the batching slice is the back half.

## Data flow (the launch pipeline)

`Planner.launch(goal, project_root, spec_path=?)` in `planner.py` runs these steps:

1. **Build context** — `build_planning_context` (`context.py`) assembles a markdown
   prompt: goal, compact CRG architecture overview, the existing Mikado graph state, a
   batching-policy note, the v2 manifest schema + instructions, and (optionally) the spec
   text. Written to `<root>/.milknado/planning-context.md`.
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

- **CRG degradation, not failure**: if CRG is unavailable, the architecture section
  prints a `_(... skipped)_` marker and planning continues. `_safe_ensure_crg` in
  `planner.py` catches CRG init failures and proceeds without structural edges.
- **Resume awareness**: `resuming = len(graph.get_all_nodes()) > 0`. The instructions
  section switches to "do NOT recreate existing nodes" and the graph section dumps
  progress counts, failed nodes (need re-planning), and ready-to-execute nodes.
- **Schema is embedded in the prompt** — `_instructions_section` ships the full v2 JSON
  schema, enum constraints, granularity guidance ("emit file-level changes, let the solver
  batch"), and the MCP-targeting note (require repository inspection for real hash anchors).
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

## Mega-batch detection (domain-owned)

`BatchPlan.mega_batch_change_count` (`domains/batching/change.py`) is the canonical
mega-batch signal: it scans **all** batches and returns the largest change count exceeding
`MEGA_BATCH_THRESHOLD = 5`, else `None`. Detection is owned by the plan producer; both
entrypoints read the property and choose their own **reaction**:

- **MCP** (`_plan_batches_impl`) raises `MegaBatchAborted` when any batch exceeds the
  threshold and `force_single_batch` is False — autonomous, fail-fast.
- **CLI** (`_plan_summary`) appends a mega-batch note to the plan summary — human-in-loop,
  non-fatal.

The CLI-reports / MCP-aborts split is intentional (milknado#90, resolved-by-design: a human
watching the CLI can eyeball the report; an autonomous MCP caller has no human, so it fails
fast). Detection is domain-owned *by construction*; reaction stays per-entrypoint policy.

Keep this distinct from `Batch.oversized` / `oversized_count` — that is a **token-budget**
signal (`DUMB_ZONE_BUDGET`), not a change count. The two must not be folded.
`MEGA_BATCH_THRESHOLD` is an uncalibrated heuristic ("one batch = one ralph loop = one
review unit"); tune it in `change.py` if calibration data appears.

**History.** This replaced a caller-shadowed guard: `check_mega_batch` lived in `planner.py`,
was promoted public in #111 only so the MCP entrypoint could reach it across the slice
boundary, and was never wired into `Planner.launch` — an encapsulation smell a diff-scoped
`/age` pass graded as clean. It also early-returned unless the plan had exactly one batch, so
a multi-batch plan holding one oversized batch passed silently. The producer-owned property
fixes both. See `history/review-lessons.md` § "Review scope: diff-scoped review can't see an
inherited encapsulation smell" and
[easy-cheese#110](https://github.com/paulnsorensen/easy-cheese/issues/110).
