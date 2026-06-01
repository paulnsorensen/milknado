# Graph Domain — Mikado Dependency Graph

The graph slice (`src/milknado/domains/graph/`) is the persistent dependency
graph at the heart of Milknado. A *goal* is the root; its prerequisites hang
below it as a DAG. The Mikado rule is inverted from intuition: **children are
prerequisites of their parent** — a node only becomes runnable once all its
children are `DONE`. Leaves run first; the root completes last.

## Core types (`src/milknado/domains/common/types.py`)

- `MikadoNode` — frozen dataclass. Identity is the autoincrement `id`. Carries
  `status`, `parent_id`, run-ownership fields (`run_id`, `pid`, `worktree_path`,
  `branch_name`), batching fields (`oversized`, `batch_index`), timestamps, and
  `kind`.
- `NodeStatus` — `PENDING | RUNNING | DONE | BLOCKED | FAILED`.
- `NodeKind` — `ROADMAP | GOAL | TASK` (default `TASK`).
- `MikadoEdge` — a `parent_id -> child_id` dependency.

## Persistence (`_persistence.py`)

State lives in a SQLite db at `.milknado/milknado.db` (gitignored; force-add to
carry across ephemeral containers). Tables: `nodes`, `edges` (composite PK,
FKs to nodes, **no** `ON DELETE CASCADE`), `file_ownership` (node → owned file
paths), `plan_state` (single-row spec hash), `batch_plans` (solver history).

`MikadoGraph.__init__` (`graph.py`) opens the connection in **WAL mode** with
`foreign_keys=ON` and `busy_timeout=5000` — the detached runner and the MCP
server both write the same db concurrently, so the busy window is explicit.
`create_tables` + `ensure_schema` run on every open: `ensure_schema` is an
additive `ALTER TABLE` migration for columns added over time (`run_id`, `pid`,
`dispatched_at`, `kind`, `completion_duration_seconds`), and `row_to_node`
defends missing columns so old rows still deserialize. `close()` runs
`PRAGMA wal_checkpoint(TRUNCATE)` so a non-last-connection close folds the WAL
tail into the main `.db` — without it a tool call's committed writes could be
lost on container reclaim before the WAL checkpoints.

## Module split

`MikadoGraph` is a thin facade over free functions, kept this way so `graph.py`
stays under the 300-line ceiling. Every function takes the `sqlite3.Connection`:

- **`graph.py`** — the `MikadoGraph` class: traversal queries, dispatch helpers,
  cycle guard, plugin notification.
- **`_transitions.py`** — the status state machine and atomic claim/release.
- **`_mutations.py`** — structural edits: subtree delete, field update, reparent,
  `would_create_cycle`.
- **`_analytics_facade.py`** — `_AnalyticsFacade` mixin; pure pass-throughs to
  `_persistence` for batch plans, completion durations, spec hash, dispatch time.
- **`traversals.py`** — `walk_ancestors`, a leaf→root single-path walk.
- **`display.py`** — rendering only.

## Status state machine (`_transitions.py`)

`VALID_TRANSITIONS` (in `types.py`) is the authority. `assert_transition`
checks the move before any write, raising `InvalidTransition` rather than
corrupting a row. Allowed edges:

- `PENDING → {RUNNING, BLOCKED, FAILED}`
- `RUNNING → {DONE, FAILED, BLOCKED, PENDING}`
- `BLOCKED → {PENDING}`, `FAILED → {PENDING}`
- `DONE → {}` (terminal — once done, never reopened)

`transition_status` sets `completed_at` on `DONE`. `mark_failed` / `mark_pending`
clear run-ownership fields (worktree/branch/run_id). Status changes fire
`_notify_status_change`, which calls registered `PluginHook.on_node_status_change`
(plugin exceptions are logged, never propagated).

### Atomic claim / reclaim / fence (the concurrency core)

`claim_node`, `release`, `mark_terminal`, and `set_pid`/`set_worktree`
(`_persistence.py`) **deliberately bypass `assert_transition`**: the SQL `WHERE`
clause *is* the guard, evaluated atomically by SQLite under the write lock,
which is correct across processes where an in-process mutex would not be.

- **`claim_node`** — one conditional `UPDATE` flipping a claimable node
  (`pending|failed|blocked`) to `running`, writing the `run_id` fence + dispatch
  timestamp + resetting `pid` to NULL. `rowcount == 1` means this caller won; a
  concurrent claimant gets 0. This is the cross-process mutual-exclusion point.
- **`run_id` is a fence**: every later ownership-gated write
  (`mark_terminal`, `release`, `set_pid`, `set_worktree`) carries
  `AND run_id = ?`. If the node was re-claimed under a new run, the stale
  owner's write hits zero rows and is silently dropped.
- The `AND status = 'running'` guard on `release` / `mark_terminal` prevents
  walking a `DONE` node backward (DONE keeps its run_id, so the fence alone
  isn't enough).
- **`try_reclaim`** (`graph.py`) frees a RUNNING node whose owner pid is
  provably dead (`_pid_alive`), short-circuiting the stale-running timeout. A
  live or pid-unknown owner is left intact.

## Mutations (`_mutations.py`)

- **`delete_subtree`** — refuses a node with edge-children unless `cascade=True`.
  Deletes the whole subtree in **one transaction** (post-order, dedup'd for
  diamonds) so a mid-delete failure rolls back rather than half-removing.
  `_delete_one` manually deletes dependent `edges`/`file_ownership` rows and
  nulls dangling `parent_id` references because the FKs lack cascade.
- **`reparent`** — assumes the **single-parent tree model**: replaces all
  incoming edges, rejects self/descendant parents (cycle).

## Invariants

- **Acyclicity** — `add_edge` and `reparent` run `would_create_cycle`
  (walks ancestors of the proposed parent; reaching the child means a loop).
- **Prerequisite (Mikado) semantics** — `get_ready_nodes` returns PENDING
  non-root nodes whose children are all DONE (or who have no children, i.e.
  leaves). `get_next_runnable(kind)` is the first ready node of a given kind.
- **Roots vs leaves** — a *root* has no incoming edge (not in `edges.child_id`);
  a *leaf* has no outgoing edge. `complete_root` auto-marks the root DONE only
  once every non-root node is DONE (it briefly transitions root through RUNNING
  since `PENDING → DONE` is not a legal direct move).
- **`parent_id` column vs `edges`** — `add_node` writes both: the denormalized
  `parent_id` for fast upward walks and the canonical `edges` row. `reparent`
  keeps them in sync. Multi-parent wiring via raw `add_edge` is possible but
  `reparent` collapses a node to a single parent.
