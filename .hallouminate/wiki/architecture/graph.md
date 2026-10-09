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
paths), `plan_state` (single-row spec hash), `batch_plans` (solver history),
`runs` (run lifecycle rows — replaced the `.milknado/runs/*.state.json`
sidecars in PR #127, closes #100), and `run_messages` (append-only worker
deposit channel, UNIQUE `(run_id, seq)` — closes #122).

`MikadoGraph.__init__` (`graph.py`) opens the connection in **WAL mode** with
`foreign_keys=ON` and `busy_timeout=5000` — the detached runner and the MCP
server both write the same db concurrently, so the busy window is explicit.
`create_tables` runs on every open but short-circuits: a `sqlite_master` probe
for the `nodes` table skips the `executescript` (one batch of `CREATE TABLE IF
NOT EXISTS` / `CREATE INDEX IF NOT EXISTS`) once the db is initialized.

The base script creates the main graph tables and indexes.
Graph initialization also runs the graph-owned `MIGRATIONS` ladder.[^graph-current-setup]
This forward-only setup creates current-schema objects on fresh databases.
It does not authorize historical-data backfills or release-compatibility transformations.[^graph-no-backfill]

`close()` runs
`PRAGMA wal_checkpoint(TRUNCATE)` so a non-last-connection close folds the WAL
tail into the main `.db` — without it a tool call's committed writes could be
lost on container reclaim before the WAL checkpoints.

### Runs repo (`runs` / `run_messages`)

Lifecycle semantics live in [[execution]]; the repo invariants live here:

- **`start_run`** INSERTs a `status='running'` row; **`finish_run`** UPDATEs to
  terminal, gated `AND status = 'running'` — **first terminal write wins**
  (commit 2d0c833). A late terminal write (wedged worker recovering after
  cancel's takeover) raises `RunFenceLostError`, never clobbers the winner.
  Dispatch callers treat that exception as an adopted terminal result. This is
  the run-level mirror of the node-level `mark_terminal` philosophy below.
- **`set_run_pid`** is gated on `status='running'` for the same reason — a
  runner that already wrote terminal state is never walked back toward running.
- **`deposit_run_message`** assigns `seq` and inserts in **one statement**
  (`INSERT … SELECT COALESCE(MAX(seq),0)+1 … RETURNING seq`) so concurrent
  depositors for the same run cannot race `MAX(seq)` into a UNIQUE collision.
  Requires SQLite ≥3.35 (`RETURNING`). It also calls `_prune_run_messages`
  unconditionally on every deposit (7-day age cutoff + 1000-row retention cap,
  both scoped to terminal runs only).
- **`runs.status` has exactly one string owner**: `_RUN_STATUS_RUNNING =
  "running"`, bound as a query parameter everywhere the column is compared
  (`start_run`, `finish_run`, `set_run_pid`, `_prune_run_messages`,
  `deposit_review_verdict`). Before this, `_prune_run_messages` inlined the
  literal as `status != 'RUNNING'` (uppercase) while every writer stored
  lowercase `'running'`; SQLite's default BINARY collation made that
  comparison case-sensitive, so the "terminal runs only" guard matched *every*
  run, including ones still executing, and `deposit_run_message`'s
  unconditional prune could delete message history out from under a
  long-running node a poller was about to read. Invisible because the wrong
  branch was the permissive one — no test exercised pruning at all until the
  fix (issue #329, commit b1713c9). The constant lives here rather than
  importing `loop._run_types.RunStatus`, to avoid putting a `domains/` module
  behind a private symbol of the vendored loop engine (see [[review-lessons]]).
- **`runs_for_node()`** lists every run for a node and does not select an owner.
  Callers that reconcile a claimed node use **`latest_terminal_run(node_id,
  run_id)`**, which requires the owning fence in its SQL predicate. A stale run
  cannot mask the owner's terminal row.
- Connections are **not cross-thread**: the async worker thread and the
  detached runner each open their own graph for the terminal write.



### Coordinator schema ownership

Fresh graph initialization creates the complete current coordinator schema before session startup.[^pr518-core-schema]
The ladder registers core sessions, links, events, and indexes in steps 41–46.
The event definition includes `operation_hash`; later session startup does not add the column.[^pr518-core-schema]
Moving the current definitions avoids an `ALTER` step against a table that does not yet exist.
The change preserves existing keys, foreign keys, indexes, and journal identity.
It adds no historical-data transformation or compatibility helper.
Fresh-graph and reopen tests verify the column, required objects, and exact current schema version.[^pr518-schema-tests]
Step 47 creates web command receipts without a session foreign key, so start commands can reserve receipts before sessions exist.
Steps 48–49 add run verification fields.
Step 50 creates coordinator plan proposals with a session foreign key, manifest, context path, revision, and status.[^pr520-current-schema]
These steps create current-schema objects. They do not transform older release data.

Steps 51–52 create native turn-launch records and the unique active-scope index.
Steps 53–54 add supervisor identity; steps 55–56 add turn and provider-session identities to journal events.
Steps 57–59 add stream keys, references, and depths; step 60 indexes stream lookup.[^pr521-current-schema]
The ladder retains steps 31–50 and creates current-schema objects before coordinator startup.
These steps add no old-data backfill, lazy schema creation, or release-compatibility transformation.[^pr521-current-schema]

### Append-only coordinator stream history

Append-only coordinator stream history compacts redacted text and rejects corrupt references while preserving exact public history.
Compact suffix rows do not rewrite earlier events.[^pr521-stream-storage][^pr521-stream-integrity]
A hash separates streams by coordinator, turn, provider session, event kind, and native event identity.[^pr521-stream-storage]
Suffix chains contain at most 32 references.
The next update writes a full checkpoint when the previous depth reaches 32.
Replacement text, rollover text, and terminal states also write full checkpoints.[^pr521-stream-storage]

History reads reconstruct each exact bounded text and preserve event sequences and cursors.
Reads can start after a checkpoint and still reconstruct the preceding reference chain.
A request-local cache avoids repeated reconstruction; reopened databases produce the same public history.[^pr521-stream-history]
The production snapshot reader calls `control_history` with a private, trusted `_HistoryRead` value.
Snapshot reads share one reconstruction cache and preserve cursor, expiry, recovery-only, and maximum-sequence rules.
Snapshot mode neither prunes events nor commits the surrounding read transaction.
Ordinary `control_history` calls retain their expiry pruning and commit behavior.[^pr521-stream-history]

Redaction runs before compaction.
The common domain owns the shared `redact_control_text` function.
Coordinator journal and public exports retain direct aliases to the same function.
Native stream diagnostics import the common public interface, which avoids a coordinator-to-session import cycle.
The ownership change preserves the redaction patterns and behavior.[^pr521-shared-redaction]
Raw tool payloads remain elided.
Only identified assistant and error events use compact stream storage.
Permission, input, terminal, and recovery records retain their public history behavior.
Diagnostic records retain their expiry rules.[^pr521-stream-redaction]
Missing references, cross-owner references, invalid depths, and forward references fail loudly.[^pr521-stream-integrity]
See [Execution and Dispatch](./execution.md#coordinator-recovery-concurrency) for native turn authority and recovery.


[^graph-current-setup]: AGENTS.md:114-122; src/milknado/domains/graph/_persistence.py:261-301.
[^graph-no-backfill]: AGENTS.md:109-122.
[^pr518-core-schema]: src/milknado/domains/graph/_coordinator_schema.py:3-61; src/milknado/domains/graph/_persistence.py:240-252.
[^pr520-current-schema]: src/milknado/domains/graph/_coordinator_schema.py:47-61; src/milknado/domains/graph/_persistence.py:244-252; tests/coordinator/test_web_receipt_schema.py.
[^pr518-schema-tests]: tests/coordinator/test_schema_setup.py:29-69.



[^pr521-current-schema]: src/milknado/domains/graph/_persistence.py (`MIGRATIONS`, `SCHEMA_VERSION`); src/milknado/domains/graph/_coordinator_schema.py (`CREATE_TURN_LAUNCHES`, `CREATE_ACTIVE_TURN_INDEX`, `CREATE_EVENTS_STREAM_INDEX`); AGENTS.md:109-122.
[^pr521-stream-storage]: src/milknado/domains/coordinator/_stream_history.py:12-14,65-73,132-156.
[^pr521-stream-history]: src/milknado/domains/coordinator/journal.py (`_HistoryRead`, `_record`, `control_history`, `snapshot_control_history`); tests/coordinator/test_turn_stream_storage.py (`test_cumulative_stream_uses_less_storage_without_changing_history`).
[^pr521-stream-redaction]: src/milknado/domains/coordinator/journal.py (`_prepared_event`, `_write_event`, `append_stream_control_event`); src/milknado/domains/coordinator/turns.py (`record_turn_event`).
[^pr521-shared-redaction]: src/milknado/domains/common/redaction.py (`redact_control_text`, `_redact_quoted`); src/milknado/domains/common/__init__.py; src/milknado/domains/coordinator/journal.py (direct redactor alias); src/milknado/domains/coordinator/__init__.py; src/milknado/loop/sessions/_stream.py (`capture_failure`).
[^pr521-stream-integrity]: src/milknado/domains/coordinator/_stream_history.py:76-129.

_Source: PR #521 current-schema, compact history, and shared redaction source contracts · Updated: 2026-10-09 · Supersedes: coordinator schema coverage ending at step 50 and shifted journal citations._


## Module split

`MikadoGraph` delegates persistence and transition operations to private modules.
These modules take the shared SQLite connection and keep the facade within its existing file-size budget.

- **`graph.py`** — the `MikadoGraph` class: traversal queries, dispatch helpers,
  cycle guard, plugin notification.
- **`_transitions.py`** — the status state machine and atomic claim/release.
- **`_connection.py`** — SQLite open, quick-check, and corrupt-database quarantine; shared by normal opening, snapshots, and recovery.
- **`_mutations.py`** — structural edits: subtree delete, field update, reparent,
  `would_create_cycle`.
- **`_analytics_facade.py`** — `_AnalyticsFacade` mixin; pure pass-throughs to
  `_persistence` for batch plans, completion durations, spec hash, dispatch time.
- **`_edge_facade.py`** — synchronized `add_edge`/`remove_edge` graph API over
  `_creation` transaction functions.
- **`traversals.py`** — `walk_ancestors`, a leaf→root single-path walk.
- **`display.py`** — rendering only.
- **`render_dot.py`** — pure Graphviz DOT renderer for the *live* graph
  (`render_dot(nodes, children_map)`, PR #362), the sibling of the wiki
  domain's roadmap-document renderer. Output is deterministic: nodes sorted by
  `id`, edges collected into a sorted deduped set (so diamonds and repeated
  wiring collapse), labels escaped for backslash/quote/control chars. Shape
  encodes `NodeKind` (`box3d`/`folder`/`box`); archived nodes render dashed in
  a grey palette. It declares its **own** `_STATUS_STYLES` status→colour map —
  value-identical to the wiki domain's map in `domains/wiki/render.py` (that
  map keys by string literal, this one by `NodeStatus.*.value`), yet
  deliberately *not* imported from it, to avoid a `graph → wiki` dependency.
  This is the same "duplicate a small constant to keep a domain boundary clean"
  call recorded for `_RUN_STATUS_RUNNING` above: the two maps coincide only
  because both render the shared `NodeStatus` vocabulary, so keep them aligned
  by intent, not by collapsing them into one import.

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

- **`claim_node`** uses `BEGIN IMMEDIATE` to serialize capacity checking and the guarded ownership update across connections.
  Capacity counts task nodes with `status='running'` and a non-null `run_id`.
  Goals, unowned status edits, and additional diagnostic run rows consume no slots.
  `open_project_graph` supplies `concurrency_limit` from project configuration.
  `ConcurrencyLimitReached(running, limit)` reports refusal without changing the task.
  `SELECT changes()` reports claim success inside the transaction; SQLite cursor `rowcount` is unreliable for this CTE-prefixed update.
  Dispatch refusal also releases newly acquired ancestor-goal claims.
  Terminal transitions or fenced release free capacity; review redispatch retains the same node ownership.
- **`run_id` is a fence**: every later ownership-gated write
  (`mark_terminal`, `release`, `set_pid`, `set_worktree`) carries
  `AND run_id = ?`. If the node was re-claimed under a new run, the stale
  owner's write hits zero rows; `set_pid` and `set_worktree` log a warning,
  while terminal and release writes report their failed fence to callers.
- The `AND status = 'running'` guard on `release` / `mark_terminal` prevents
  walking a `DONE` node backward (DONE keeps its run_id, so the fence alone
  isn't enough).
- **`nodes.run_id` is a column, not an FK**: a DONE-node re-run via the sync
  MCP tool overwrites the fence with a freshly minted run_id that has no `runs`
  row — intentional and harmless, because the fence is only consumed during
  RUNNING-node reconciliation and a later re-dispatch overwrites it with a
  tracked row (reviewed and kept in PR #127).
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
- **Display roots** — `project_graph` selects nodes whose `parent_id` is absent from the snapshot node map, including `None` and missing parents. It preserves snapshot node order. Display roots therefore differ from domain roots based on incoming edges; `snapshot.root_ids` does not drive this projection.[^display-roots]

[^display-roots]: `src/milknado/app/graph_view.py:52-72`; `src/milknado/app/graph_panels.py:78-86`.

_Source: PR #488 atomic admission; graph projection root cleanup; `graph/_transitions.py:149-202`, `graph/graph.py`, `project.py`, and `tests/test_execution_admission.py` · Updated: 2026-09-30 · Supersedes: claim-only concurrency without shared capacity enforcement._
