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

1. `WorktreeManager.ensure_clean` removes any stale worktree tracked for the
   node — fail-closed: a refusal (dirty/unlanded orphan) is logged with what is
   at risk and the orphan is kept (see "Fail-closed worktree teardown" below).
2. Slugify description → worktree path from `config.worktree_pattern`; **reject
   paths that resolve outside `project_root`** (path-escape guard).
3. If the canonical path is still occupied (a preserved orphan),
   `_relocate_occupied` degrades: deterministic `-<n>` suffix on both path and
   branch, first free slot wins — a refused orphan never blocks dispatch.
4. Create the worktree on branch `milknado/{id}-{slug}` (or the relocated
   `…-{n}` variant).
5. Two claim paths diverge here (see "Two claim paths" below): set the node
   RUNNING (or attach worktree metadata to an already-claimed node), generate
   `RALPH.md` via `_create_ralph_run`, start the ralph run, record `dispatched_at`.

On any failure inside the try, `_cleanup_failed_dispatch` runs a **fenced**
state reset (release under the node's `run_id` if it has one, else `mark_pending`)
then discards the worktree. The fence stops a failed dispatch from walking a node
back to PENDING after a different run already re-claimed it.

`_create_ralph_run` renders the shared `dispatch.brief.render_brief` output, wraps it in ralph-only loop scaffolding, writes `RALPH.md`, creates and starts the ralph run, and returns its `run_id`.

## Run identity — runs are per-task-dispatch; there is NO coordinator-run entity

The single most important fact about the run model, because designs keep
assuming otherwise: **a milknado "run" is one task dispatch, not a coordinator
session.** `make_run_id(node_id)` embeds the *task node's* id
(`node-<id>-<UTCstamp>-<8hex>`, `_runstate.py`), and every dispatch tool mints a
fresh one per dispatched task. No identity in the system spans the sibling
dispatches a coordinator makes while driving a goal. The only thing that does
span them is the MCP server process itself (one per worktree) — i.e. the pid.

Why this matters: the worktree-shared-graph spec (PR #131) said goal claims are
"owned by that run/session". That entity doesn't exist, and the first
implementation wired claims literally — fresh `make_run_id` per dispatch — so
the first task dispatched under a goal fenced out its own siblings (review
blocker). The shipped fix anchors the same-coordinator exemption on
`claim["pid"] == os.getpid()` because the server process IS the coordinator
under the current one-process-per-worktree model.

**Rule for future specs:** any design that says "owned by the run/session" must
name which concrete identity it means — task-dispatch run_id (per-task, never
spans siblings), server pid (spans siblings, dies with the process), or a new
coordinator-run entity (doesn't exist yet; would need creating). Don't let
"run/session" pass spec review unexamined.

## Goal-claim fencing (`goal_claims`, PR #131)

Cross-worktree coordinator fencing, layered above per-node `claim_node`: all
three dispatch tools claim the closest ancestor goal before dispatch and refuse
when it's claimed by a different **live** process. Design decisions and their
why:

- **`INSERT OR IGNORE` with pid written atomically in the same statement** is
  the whole mutex (`claim_goal_row`, `_goal_claims.py`). The pid must ride in
  the INSERT: a two-step insert-then-set-pid left a window where a NULL-pid
  claim from a healthy coordinator could be "reclaimed" by a racing one
  (pass-2 review high).
- **Same-pid exemption** (the `claim["pid"] != owner_pid` guard in
  `claim_ancestor_goal_for_dispatch`, `domains/graph/graph.py`, over
  `ancestor_goal_claimed_by_other` in `_goal_claims.py`): a claim held by the
  current process doesn't fence sibling dispatches. See the run-identity section
  above for why pid, not run_id.
- **Release fires on goal-node terminalization only** (`mark_done` /
  `mark_failed` / `mark_terminal` → `release_goal_claim_on_terminal`). A
  cancelled run that never terminalizes its goal leaves a claim row, but
  pid-liveness reclaim clears it at the next dispatch attempt once the process
  dies — soft leak, self-healing, no stale-timeout wait.
- **Intentionally unwired seams**: `ancestor_goal_claimed_by_other`'s
  `caller_run_id` parameter and the `release_goal` method have zero production
  callers. They are the hooks for a future coordinator-run entity (multi-run
  servers / respawning coordinators). If that entity ever becomes real, the
  same-pid exemption stops being a faithful proxy and claim ownership must move
  to the threaded run identity — do not delete these seams as dead code.
- **pid reuse fails safe**: a recycled pid keeps a stale claim looking alive →
  spurious refusal, never a false-allow. Same pre-existing exposure as the
  node-claim path; accepted.

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
  rebases onto `feature_branch`, and tears down the worktree in a `finally` —
  but only when the rebase reported success; a conflicting rebase or in-flight
  exception keeps the worktree (that work is by definition unlanded).
  `RebaseAbortError` re-raises (repo corruption — never swallowed); other
  exceptions become a failed `RebaseResult`. On success → `_mark_terminal(DONE)`
  and records completion duration; on failure → `_mark_terminal(FAILED)` + build a
  `RebaseConflict`. Newly dispatchable nodes are recomputed only on success.

`_mark_terminal` is fenced: with a `run_id` it calls `graph.mark_terminal(id,
run_id, status)` (atomic `WHERE run_id=? AND status='running'`) and returns
whether the write landed; with no `run_id` (legacy/test) it transitions
unconditionally. Duration is recorded only when the write actually landed.

## Fail-closed worktree teardown

Teardown refuses to destroy work by default; destruction is an explicitly
named act. `GitAdapter.remove_worktree(path, target="HEAD")` is fail-closed:

- **Dirty guard** — `git status --porcelain` pre-check (plus git's own native
  no-`--force` refusal as backstop); dirty files are named in the diagnostic.
- **Landed check** — a single git-native probe, no `gh`/PR dependency:
  `git merge-base --is-ancestor <worktree_head> <target>`, exact for milknado's
  own squash → rebase → `merge --ff-only` land path (the landed HEAD equals the
  target tip). A non-ancestor or inconclusive probe **refuses** — never destroy
  on a guess.
- Refusal raises `UnlandedWorkError` naming the worktree and the at-risk work
  (dirty files and/or the unlanded commit range).
- `GitAdapter.force_remove_worktree` is the only `--force` teardown, reachable
  solely via `WorktreeManager.discard` (asserted by a source-audit test in
  `test_adapters_git.py`).

Refusal semantics per call site:

| Site | On refusal |
|---|---|
| `WorktreeManager.remove` (from `rebase_and_merge` / `Executor.fail`) | **hard-fail** the caller — the old warn-and-swallow after a `--force` remove was silent destruction |
| `cancel.py:_reconcile_cancel` (routes through `WorktreeManager.remove`, never the raw adapter) | **hard-fail** the cancel; worktree and node preserved |
| `WorktreeManager.ensure_clean` (pre-dispatch cleanup) | **degrade** — log what is at risk, keep the orphan, dispatch relocates |
| Orphan prune in `milknado_run_loop_start` (`mcp/ralph.py`) | **degrade** — same; the run loop is never blocked |
| `WorktreeManager.discard` | unchanged — this IS the explicit destructive path |

`rebase_and_merge`'s landed check runs against `feature_branch` and the
worktree's HEAD *at removal time* — post-squash/rebase/ff — so the happy path
removes cleanly while a squashed-but-never-fast-forwarded HEAD (the old
silently-destroyed case) refuses. Non-refusal removal failures (nothing was
destroyed; the worktree is still on disk) stay warn-and-swallow so the node
lifecycle keeps moving.

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



### Ralph-loop pipe ownership

A reader thread owns each `Popen` stdout or stderr stream until EOF.[^pipe-ownership]
Cleanup must not call `os.close(stream.fileno())` while the Python stream remains open.
A detached descendant can inherit a pipe and keep its reader alive after the direct child exits.
Cleanup therefore bounds the join and leaves each live reader with its stream.
The reader closes the stream after EOF or a terminal read error.
Raw descriptor closure leaves a stale `TextIOWrapper`.
Later finalization can close an unrelated descriptor after the operating system reuses the descriptor number.

[^pipe-ownership]: `src/milknado/loop/_agent.py:878-899,927-976`; `tests/loop/test_agent.py:1835-1884`

## Deposit channel — worker → coordinator results (#122)

The log-tail `summary` is lossy: a worker's complete deliverable rarely survives
a 2 KB tail. PR #127 adds a durable return channel — `run_messages`, an
append-only per-run message table (seq assigned atomically in a single
`INSERT … SELECT MAX(seq)+1 … RETURNING` statement, so concurrent depositors
cannot collide). The MCP tool `milknado_deposit_result(run_id, payload)` writes
`role='result'` rows; `milknado_run_inline_poll` returns the latest one under
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

Five coordinator-facing MCP tools drive runs: `milknado_run_inline` (blocking),
`milknado_run_inline_start` / `milknado_run_inline_poll` (async in-process worker),
and `milknado_run_loop_start` / `milknado_run_loop_poll` (detached
worktree-isolated ralph loop). All five return the **unified superset schema**
`RunDict` (`mcp/_core.py`): `run_id, node_id, status, exit_code, timed_out,
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
  `reconcile_node_status` orphan recovery uses) and tear down the worktree
  through `WorktreeManager.remove` — the fail-closed path (see *Fail-closed
  worktree teardown*): dirty/unlanded work hard-fails the cancel with
  `UnlandedWorkError` — then `git worktree prune`. No-ops cleanly when the run
  is already terminal.

## Worker-dispatch families — single-shot (3) vs ralph (4), and why both exist

> **Terminology caution.** "Family 3 / Family 4" is *our shorthand for the two
> dispatch mechanisms*, not a code symbol. In the code, `family` means the
> **agent vendor** — `claude` / `codex` / `gemini` / `cursor`
> (`DEFAULT_PLANNING_AGENT_BY_FAMILY`, the `WORKER_ALLOWED_TOOLS` keys;
> `domains/common/agent_argv.py`). Don't conflate the two.

- **Family 3 = single-shot headless worker** — `milknado_run_inline` (blocking,
  `mcp/run.py:40`) plus `milknado_run_inline_start` / `_poll` (async in-process,
  `mcp/run.py:80` / `:121`). One worker pass, brief piped on stdin, no quality
  gates. Isolation is **safe by default**: `worktree=ISOLATE` (the default) runs
  the worker in a fresh worktree+branch and rebase-merges it back on exit 0;
  `worktree=THIS_BRANCH` opts into the shared working tree (see
  [run-inline-isolation](./run-inline-isolation.md)). Chain: `milknado_run_inline` →
  `dispatch_node_sync` (`dispatch/lifecycle.py:37`) → `run_headless`
  (`dispatch/runner.py:261`) → `_execute` → `_spawn_worker` →
  `subprocess.Popen(stdin=PIPE)` + `proc.communicate(brief, timeout)` (blocks).
  The async variant swaps the blocking call for a daemon
  `threading.Thread(_async_worker)` (`dispatch/async_run.py:119`) that dies with
  the server. Mechanics detailed above in *Subprocess workers & run-state*.
- **Family 4 = ralph iterate-until-gates loop** — `milknado_run_loop_start` /
  `_poll` (`mcp/ralph.py:62` / `:89`). Detached
  `Popen([... _ralph_node_runner ...], start_new_session=True)`, no stdin, runs
  in its **own git worktree+branch**, loops until `quality_gates` pass
  (`run_node_to_completion`, `execution/headless.py:49`), then rebase-merges
  back. Refuses to start if `profile.quality_gates is None`. Survives a server
  restart (detached, pid in SQLite); poll is read-only. Mechanics in
  *Headless single-node loop* above.

**Why Family 3 exists (rationale the code doesn't state):** to dispatch a node
to a **different harness than the one orchestrating** and **block on the
single-shot result without polling**. The coordinator (say Claude Code)
overrides `worker_cmd` to point at another agent — e.g. a local-model-backed CLI
for cheap or offline nodes — and `milknado_run_inline` blocks until that foreign
worker exits, returning the result inline. No worktree, no gate loop, no poll
cycle: it is the "shell out to another model and wait" path. `worker_cmd` is the
cross-harness lever — it defaults to `profile.execution_agent` but the caller
overrides it per dispatch (`mcp/run.py:40`, docstring). **Constraint:** the
override's executable *basename* must be one of `{claude, codex, cursor-agent,
gemini}` (`validate_worker_argv` / `_ALLOWED_WORKER_EXECUTABLES`,
`agent_argv.py`) — a "local LLM" is reachable only when fronted by one of those
four agent CLIs (a family CLI pointed at a local endpoint/model), not as an
arbitrary command.

**Why Family 4 is the other shape:** when the coordinator wants to hand off the
*whole* task — "iterate until your gates pass, merge it back, tell me when
done" — and not babysit it. The caller drives no loop and owns no retries; the
detached runner does. Worktree isolation lets many ralphs run in parallel
without trampling each other; detachment lets the loop outlive the MCP server.

**Coordinator vs worker — who may call these (verified against the allowlist):**
both run-dispatch families are **coordinator-facing**; *neither* is granted to
spawned workers. `WORKER_ALLOWED_TOOLS["claude"]` (`agent_argv.py`) gives a
worker only two milknado MCP tools — `milknado_track_follow_up` and
`milknado_deposit_result` — never the run-dispatch tools. Family 4 additionally
carries a **hard, permanent prohibition**: *"COORDINATOR-ONLY: never add these
to WORKER_ALLOWED_TOOLS"* (`mcp/ralph.py:1-12`), because a worker that could
start sub-ralph-loops would recursively fork worktrees. (A casual read of the
code can mis-state Family 3 as worker-allowed — it is not; re-check
`WORKER_ALLOWED_TOOLS` before asserting otherwise.)

| | Family 3 — single-shot | Family 4 — ralph |
|---|---|---|
| Tools | `milknado_run_inline` / `_start` / `_poll` | `milknado_run_loop_start` / `_poll` |
| Process | blocking caller, or in-process daemon thread | detached subprocess (`start_new_session=True`) |
| Brief delivery | piped on stdin | node/run id via env + CLI args (no stdin) |
| Iterations | one pass | loop until quality_gates pass / timeout |
| Working tree | own worktree+branch by default (`THIS_BRANCH` opts into shared) | own worktree+branch, rebase-merge on success |
| Quality gates | none | required (refuses if unset) |
| Survives server restart | no | yes (pid in SQLite) |
| Loop / retries owned by | caller | detached runner |
| Granted to workers? | no (coordinator-facing) | no — explicit permanent prohibition |
| Reach for it when | block on a (possibly foreign-harness) single shot, no poll | hand off the whole gated loop, walk away |

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

## Tmux run substrate — opt-in, per dispatch (`adapters/tmux.py`, `dispatch/tmux_run.py`)

Both long-lived dispatch families accept `use_tmux: bool = False`
(`milknado_run_loop_start`, `milknado_run_inline_start`). The default detached /
in-process paths are unchanged when it is not passed; a dispatch parameter (not
a `milknado.toml` key) was chosen as the smallest opt-in surface. With
`use_tmux=True` the run executes inside a named tmux window and
`milknado attach <run_id>` drops you into it.

**Topology & naming contract.** One tmux session per project
(`milknado-<sanitized root dirname>`, `session_name_for`), one window per run,
window name = the full `run_id`. The target is **derived from the run_id at
read time** — no `runs`-table column. All targeting uses tmux's `=` exact-match
prefix (`=session:=run_id`); bare names fall back to prefix/glob matching (man
tmux, TARGET SPECIFICATIONS). A window-name collision at dispatch is a hard
error, never reuse.

**Fail-closed.** `ensure_tmux_ready` runs *before* the node claim: if tmux was
requested but the binary is missing or the server can't start, the dispatch
raises with a clear message — no silent fallback to the detached path. When
tmux is not requested, nothing tmux-related runs at dispatch.

**Pane = process group.** The window wrapper (POSIX sh; the milknado session's
`default-shell` is pinned to `/bin/sh` because a zsh default-shell breaks the
wrapper via `=word` expansion) runs the same runner argv the detached path
would spawn — through `env -i <allowlisted vars>` for exact parity with
Popen's replacement env, since a pane otherwise inherits the tmux *server's*
environment and would leak user secrets the worker-env allowlist strips — tees
output to both the pane and the run log the poll tools tail, and records the
runner's exit code in `.milknado/runs/<run_id>.rc`. The pane pid is recorded
as the run's pid (both families — the run-inline waiter persists it via
`set_run_pid` because a pane, unlike the in-process subprocess, survives an
MCP-server restart), so **pid-liveness stays the sole authority for
graph-state transitions** (`try_reclaim`, stale sweeps) and
`milknado_run_cancel`'s `killpg` keeps working; pane liveness is additive only
(attach precondition + diagnostics). Killing the window kills the pane's
process group — a run is never orphaned by `kill-window`. The run-inline path
stages the brief at `.milknado/runs/<run_id>.brief` (stdin redirect; a tmux
pane has no stdin pipe) and waits on the pane pid with the same
cancel-sentinel + timeout contract as `_execute_cancellable`
(`execute_in_window`).

**Window lifecycle.** `remain-on-exit on` is set from inside the pane before
the runner starts. A run that exits 0 kills its own window; a failed run's
window is preserved as a dead pane for inspection. Reconciliation is per-row
and lazy, matching the reclaim model above: when a poll observes a `done` run,
`reconcile_run_window` kills a straggler window via one exact-match query per
run row (`cleanup_run_window`) — never a pattern sweep of the session's
windows. The runs table is the expected set; tmux is consulted per row.

**Attach.** `milknado attach <run_id>` (`cli/run.py`) resolves preconditions
via `resolve_attach_target` — unknown run, finished run, missing tmux binary,
and non-tmux run each fail with a distinct message — then execs
`tmux select-window -t =sess:=run \; attach-session` (or `switch-client` when
already inside tmux).

## Shared brief and RALPH.md

`brief.render_brief` is the shared node-context markdown for native/MCP and
ralph workers: goal context, completed prerequisites, owned files, specs, and
instructions. Ralph dispatch embeds that exact brief in `RALPH.md` and adds only
the iteration-specific findings, gates, follow-up protocol, and completion
sentinel.

## Key files

- `src/milknado/domains/execution/executor.py` — `Executor`, `WorktreeManager`, dispatch/complete state machine, fencing.
- `src/milknado/domains/execution/headless.py` — `run_node_to_completion`.
- `src/milknado/domains/dispatch/brief.py` — `render_brief` (shared worker context and result-deposit instructions).
- `src/milknado/adapters/loop.py` — RALPH.md loop scaffolding.
- `src/milknado/domains/dispatch/runner.py` — subprocess spawn, async worker, cancel, orphan recovery, `reconcile_node_status`.
- `src/milknado/domains/dispatch/_runstate.py` — run-id format, log tail, cancel sentinel (run *state* lives in the SQLite `runs` table).
- `src/milknado/adapters/tmux.py` — `TmuxAdapter`, `RunWindow`, exact-match targeting, the POSIX-sh window wrapper.
- `src/milknado/domains/dispatch/tmux_run.py` — `ensure_tmux_ready` (fail-closed), `execute_in_window` (run-inline pane waiter), `cleanup_run_window` / `reconcile_run_window` (per-row lifecycle), `resolve_attach_target`.
- `src/milknado/domains/common/agent_argv.py` — `WORKER_ALLOWED_TOOLS` (per-vendor worker tool allowlist), `_ALLOWED_WORKER_EXECUTABLES` / `validate_worker_argv` (worker-cmd basename gate), `resolve_*_agent_command`.
- `src/milknado/domains/graph/_persistence.py` — `runs` / `run_messages` repo (`start_run`, `finish_run`, `deposit_run_message`) and the `goal_claims` table schema.
- `src/milknado/domains/graph/_goal_claims.py` — goal-claim repo + fencing helpers (`claim_goal_row`, `release_goal_row`, `ancestor_goal_claimed_by_other`, `claim_or_reclaim_goal`); `MikadoGraph.claim_ancestor_goal_for_dispatch` (`graph.py`) is the dispatch-time entry point.
- `src/milknado/mcp/run.py` — `milknado_run_inline*` (Family 3), `milknado_run_list`, `milknado_run_cancel`, `milknado_deposit_result`.
- `src/milknado/mcp/ralph.py` — `milknado_run_loop_start` / `_poll` (Family 4, COORDINATOR-ONLY).
- `src/milknado/mcp/_core.py` — `RunDict` unified run-result schema, the shared `FastMCP` instance, and the `resolve_project_root` / `open_graph` / status-kind-flavor parsers. (The goal-claim fencing this module once held now lives on `MikadoGraph.claim_ancestor_goal_for_dispatch`; see `domains/graph/`.)
