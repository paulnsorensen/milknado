# Execution & Dispatch — Parallel Ralph Loops

How a solved batch of ready Mikado nodes is dispatched into isolated git
worktrees, run as parallel ralph loops, and merged back. Two domains cooperate:
`execution/` owns the per-node worktree lifecycle and terminal state machine;
`dispatch/` owns subprocess workers, run-state rows, and node-status
reconciliation.

## What runs in parallel

A "ralph loop" is one node executing in its own worktree against a freshly
generated `RALPH.md`. The batcher (`domains/batching`) groups ready nodes into
batches that are safe to run together; within a batch each node is an
independent ralph loop. `get_dispatchable_nodes` (executor.py) is the in-flight
gate: it takes `graph.get_ready_nodes()`, runs `check_parallel_safety` to drop
nodes that share owned files with another ready node (the later id is blocked,
logged, deferred), and returns the safe set. File-ownership conflict is the only
parallelism constraint — dependencies are already encoded in readiness.

## Dispatch lifecycle (`Executor`, executor.py)

`dispatch(node_id, config)` wraps `_dispatch_once` in a transient-retry loop
(`dispatch_max_retries`, exponential backoff). `InvalidTransition`/`ValueError`
re-raise immediately; transient failures (OSError, timeout, 429/rate-limit,
exit 124/137/143 — see `_is_transient`) retry.

`_dispatch_once`:
1. `WorktreeManager.ensure_clean` removes any stale worktree tracked for the node.
2. Slugify description → worktree path from `config.worktree_pattern`; **reject
   paths that resolve outside `project_root`** (path-escape guard).
3. Create the worktree on branch `milknado/{id}-{slug}`.
4. Two claim paths diverge here (see "Two claim paths" below): set the node
   RUNNING (or attach worktree metadata to an already-claimed node), generate
   `RALPH.md` via `_create_ralph_run`, start the ralph run, record `dispatched_at`.

On any failure inside the try, `_cleanup_failed_dispatch` runs a **fenced**
state reset (release under the node's `run_id` if it has one, else `mark_pending`)
then discards the worktree. The fence stops a failed dispatch from walking a node
back to PENDING after a different run already re-claimed it.

`_create_ralph_run` builds node context (`_context.build_node_context`: goal +
why-chain from `walk_ancestors`, owned files, and CRG impact radius — CRG
failures degrade gracefully to a skipped section), writes `RALPH.md`, creates
and starts the ralph run, returns its `run_id`.

## Two claim paths (the RUNNING-RUNNING gotcha)

`_dispatch_once` branches on `already_claimed = node.status == RUNNING and
node.run_id is not None`:

- **Ralph/MCP detached path** — the dispatching parent claims the node RUNNING
  under a `run_id` *before* the detached runner reaches `_dispatch_once`. Re-marking
  RUNNING would be an illegal RUNNING→RUNNING transition that kills the detached
  run at startup. Instead it calls `set_worktree(node_id, node.run_id, ...)`
  (fence-gated) and must **not** clobber the existing `run_id` — `set_pid` and the
  completion fence both depend on it.
- **In-process TUI / e2e path** — node arrives still PENDING; normal
  `mark_running` → `set_run_id`.

This split is why every terminal write is fenced on `run_id` rather than just
checking status.

## Completion & idempotency (commit a994fed)

`Executor.complete(node_id, feature_branch)`:

- **Terminal node short-circuit** — if the node is already DONE/FAILED, return a
  synthesized result *before* `rebase_and_merge`. This is the idempotency fix:
  `mark_done` leaves `worktree_path` set, so without the guard a duplicate
  completion (same run reporting done twice, or a re-run) would re-run
  squash/rebase/worktree-removal. First completion wins; mirrors
  `reconcile_node_status`.
- **Non-RUNNING, non-terminal (PENDING/BLOCKED)** — raise `InvalidTransition`.
  Failing loud here is deliberate: silently no-op'ing would hide a real
  state-machine bug and print a false completion in the run loop.
- **RUNNING** — `WorktreeManager.rebase_and_merge` squash-commits the worktree,
  rebases onto `feature_branch`, and removes the worktree in a `finally`.
  `RebaseAbortError` re-raises (repo corruption — never swallowed); other
  exceptions become a failed `RebaseResult`. On success → `_mark_terminal(DONE)`
  + record completion duration; on failure → `_mark_terminal(FAILED)` + build a
  `RebaseConflict`. Newly dispatchable nodes are recomputed only on success.

`_mark_terminal` is fenced: with a `run_id` it calls `graph.mark_terminal(id,
run_id, status)` (atomic `WHERE run_id=? AND status='running'`) and returns
whether the write landed; with no `run_id` (legacy/test) it transitions
unconditionally. Duration is recorded only when the write actually landed.

## Headless single-node loop (headless.py)

`run_node_to_completion` is the TUI-free twin of one `RunLoop` iteration — the
`rich.live.Live` display + keyboard thread in `RunLoop.run()` can't live in a
server/subprocess. Flow: refuse to dispatch onto an empty/`"HEAD"` branch
(detached HEAD isn't a valid rebase target — mark the node failed for parity) →
`dispatch` → `ralph.wait_for_next_completion({run_id}, timeout)` → on
timeout/non-completion `stop_run` + `executor.fail` → otherwise `complete`.
Success requires both run completion AND a clean rebase.

## Subprocess workers & run-state (runner.py, _runstate.py)

`run_headless` (blocking, via `_execute`) and `start_headless_async` (detached
thread, via `_execute_cancellable`) spawn a worker through `_spawn_worker`:
brief piped to stdin, combined stdout/stderr to a log file, cwd = worktree.

- **Worker allowlist** — `_validate_worker_argv` checks the *basename* of argv[0]
  against `{claude, codex, cursor-agent, gemini}`, defeating both prefix tricks
  (`claude-evil`) and absolute paths. Guards the explicit MCP arg, the
  `$MILKNADO_WORKER_CMD` env, and the default `claude -p`.
- **Env scrubbing** — `_build_worker_env` passes only an allowlist of system vars
  plus `MILKNADO_*`. INVARIANT: no `MILKNADO_*` var may hold a secret — every one
  is forwarded verbatim. API keys / tokens / DB URLs stay in the parent.

Run-state lives in the **SQLite `runs` table** (PR #127, closes #100) — schema
and repo functions in [[graph]]. Only **log files and cancel sentinels** remain
on the filesystem under `.milknado/runs/`; the JSON `.state.json` sidecars are
gone, as a clean cut with no migration or sidecar-import shim (pre-release "No
Migration Code" rule). `run_id` format: `node-<id>-<UTCstamp>-<8hex>`; the
4-byte suffix (not 2) is required because back-to-back runs of one node share a
wall-clock second and only the suffix distinguishes their ids.

Lifecycle: `start_run` INSERTs a rescuable `running` row **before**
spawning/blocking — if the client times out and the server is killed mid-run,
the terminal write never lands, and without that early row
`fail_stale_running_runs` could not release the node. The worker runs, then
`finish_run` UPDATEs to terminal. `finish_run` is gated
`AND status = 'running'` — **first terminal write wins** (commit 2d0c833): a
wedged worker that recovers after cancel already took over the terminal write
cannot clobber `cancelled`/`failed` back to `done`, and vice versa; a dropped
late write is logged, never silent. Both cancel takeover paths re-read the row
after writing, so the run's *actual* terminal status — not the canceller's
assumption — is what reconciles the node.

### Cancellation (cooperative, not signals)

The async worker shares the MCP server's process group, so `killpg` would kill
the server. Instead `request_cancel` drops a `<run_id>.cancel` sentinel
(atomic write); `_execute_cancellable` polls it every `_CANCEL_POLL_SECS`,
SIGTERMs its own process, waits `_CANCEL_GRACE_SECS`, then SIGKILLs.

**The worker owns the terminal write for a cancelled run** — `run_cancel` only
requests cancellation. Finalizing in `_async_worker` (not in the cancel call)
closes the state-clobber race the old signal-then-overwrite path left open, and
the storage layer now enforces the same invariant from the other side:
`finish_run`'s running-gate drops the loser of the race in either direction.
The sentinel is cleared in a `finally` after the terminal write so a reused run
dir never carries a stale cancel into the next run.

The brief is written from a daemon thread (`_write_worker_stdin`) so a brief
larger than the OS pipe buffer can't block before the cancel/timeout poll loop
starts. `_async_worker`'s except branch guarantees a terminal "failed" write on
any spawn/write failure — otherwise the run row sticks on "running" and locks
the node out forever.

## Deposit channel — worker → coordinator results (#122)

The log-tail `summary` is lossy: a worker's complete deliverable rarely survives
a 2 KB tail. PR #127 adds a durable return channel — `run_messages`, an
append-only per-run message table (seq assigned atomically in a single
`INSERT … SELECT MAX(seq)+1 … RETURNING` statement, so concurrent depositors
cannot collide). The MCP tool `milknado_deposit_result(run_id, payload)` writes
`role='result'` rows; `milknado_todo_run_poll` returns the latest one under
`result` alongside the log-tail `summary`.

**The brief contract is the root fix, not the storage**: the worker brief
(`brief.py`) mandates depositing the complete deliverable before finishing,
because storage alone cannot recover a deliverable the worker never restated.

`MILKNADO_RUN_ID` is injected into the worker env **only when a `running` run
row exists**. A DONE-node re-run inserts no row (`start_run` is gated on the
node actually transitioning), so the worker gets no run_id and the mandated
deposit soft-no-ops instead of raising "run not found".

Progress messaging and coordinator→worker inbound are deliberately NOT built;
the `run_messages` shape permits them later (YAGNI — spec non-goal).

## MCP run surface — five tools, one schema (#82, #83, #107)

Five coordinator-facing MCP tools drive runs: `milknado_todo_run` (blocking),
`milknado_todo_run_start` / `milknado_todo_run_poll` (async in-process worker),
and `milknado_ralph_run_start` / `milknado_ralph_run_poll` (detached
worktree-isolated ralph loop). All five return the **unified superset schema**
`RunDict` (`_mcp_core.py`): `run_id, node_id, status, exit_code, timed_out,
rebased, log_path, summary`, every field nullable where it doesn't apply
(`summary` is None until a poll tails the log; `rebased` is None for non-ralph
runs; the start tools return `exit_code`/`timed_out` None). One client code
path handles every run type — fixing signature finding S5, where three
divergent dict shapes used to force per-tool branching. `state_path` was
dropped from the schema with the sidecars (PR #127).

Both poll tools **derive the log path from the regex-validated run_id**
(`runs_dir / f"{run_id}.log"`) rather than trusting the stored `runs.log_path`:
the db is user-editable, and a tampered path must not turn a poll into an
arbitrary-file read.

Two management tools close the lifecycle (the structural fix the run-lifecycle
bugs #38/#39/#50 pointed at):

- `milknado_run_list(project_root="", limit=50)` — enumerate recent runs from
  the `runs` table, newest first by `started_at`, bounded by `limit` so read
  cost stays flat as run history grows (an indexed query, where the sidecar
  model had to glob and stat the runs dir).
- `milknado_run_cancel(run_id)` — validates the `run_id` against `RUN_ID_RE`,
  then forks on run type. A **detached-ralph** run has its own process group, so
  `os.killpg(SIGTERM)` is safe and cancel finalizes the terminal state directly
  (`_cancel_pid_run`). An **async-headless** run shares the server's process
  group, so cancel writes the cooperative sentinel and waits a bounded window
  (`_CANCEL_FINALIZE_TIMEOUT_SECS`) for the worker to own the terminal write,
  taking over only if the worker never responds (`_cancel_async_run`). Both paths
  reconcile the node fenced on `run_id` (`_reconcile_cancel` → the same
  `reconcile_node_status` orphan recovery uses) and prune the worktree
  (`remove_worktree` + `git worktree prune`). No-ops cleanly when the run is
  already terminal.

## Orphan recovery & reconciliation

A worker can vanish (server crash) leaving a `running` run row and a RUNNING
node. `reconcile_orphan_node` is the shared three-call recovery:

1. `fail_stale_running_runs` — flip `running` rows older than
   `timeout + _STALE_GRACE_SECONDS` (30s) to `failed` — **skipping (and
   logging) a row whose recorded pid is still alive** (#54), so a slow live
   worker is never force-failed by the sweep.
2. `runs_for_node` (+ optional `run_id` fence pushed into the SQL) →
   `latest_terminal_run` (max by `ended_at`). **Fence before picking latest**:
   a stale run with a later `ended_at` could otherwise mask the current owner's
   row.
3. `reconcile_node_status` — fenced terminal transition. With a `run_id` it uses
   the atomic fenced `mark_terminal` (closes the TOCTOU where two reconcilers both
   pass a Python-level run_id check but only the matching UPDATE lands); with no
   run_id it falls back to the unconditional transition, touching only a still-RUNNING node.

## Brief vs context — two artifacts

Don't confuse them. `brief.render_brief` (dispatch) is the markdown piped to a
**subprocess worker's stdin** — goal context, done prerequisites, owned files,
and instructions (including registering follow-ups via `milknado_track_follow_up`
and depositing the final deliverable via `milknado_deposit_result`).
`_context.build_node_context` (execution) is the context **embedded in
`RALPH.md`** for a ralph run, including CRG impact radius. Headless/MCP workers
get a brief; ralph loops get RALPH.md.

## Key files

- `src/milknado/domains/execution/executor.py` — `Executor`, `WorktreeManager`, dispatch/complete state machine, fencing.
- `src/milknado/domains/execution/headless.py` — `run_node_to_completion`.
- `src/milknado/domains/execution/_context.py` — `build_node_context` (RALPH.md context).
- `src/milknado/domains/dispatch/runner.py` — subprocess spawn, async worker, cancel, orphan recovery, `reconcile_node_status`.
- `src/milknado/domains/dispatch/_runstate.py` — run-id format, log tail, cancel sentinel (run *state* lives in the SQLite `runs` table).
- `src/milknado/domains/dispatch/brief.py` — `render_brief` (worker stdin, mandates the result deposit).
- `src/milknado/domains/graph/_persistence.py` — `runs` / `run_messages` repo (`start_run`, `finish_run`, `deposit_run_message`).
- `src/milknado/mcp_run.py` — `milknado_todo_run*`, `milknado_run_list`, `milknado_run_cancel`, `milknado_deposit_result`.
- `src/milknado/mcp_ralph.py` — `milknado_ralph_run_start` / `_poll` (COORDINATOR-ONLY).
- `src/milknado/_mcp_core.py` — `RunDict` unified run-result schema.
