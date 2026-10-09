# Execution & Dispatch — Parallel loops

How a solved batch of ready Mikado nodes is dispatched into isolated git
worktrees, run as parallel loops, and merged back. Two domains cooperate:
`execution/` owns the per-node worktree lifecycle and terminal state machine;
`dispatch/` owns subprocess workers, run-state rows, and node-status
reconciliation.

## What runs in parallel

A "loop" is one node executing in its own worktree against a freshly
generated `LOOP.md`. The batcher (`domains/batching`) groups ready nodes into
batches that are safe to run together; within a batch each node is an
independent loop. `get_dispatchable_nodes` checks readiness and file-ownership conflicts.
The graph claim transaction enforces the shared project concurrency limit.
CLI, detached, inline, and native dispatch use this same admission authority.
One owned running task consumes one slot, even when it has parent and worker run records.
Capacity refusal leaves the task unchanged and starts no worker or worktree.

## Dispatch lifecycle (`Executor`, executor.py)

`dispatch(node_id, config)` wraps `_dispatch_once` in a transient-retry loop
(`dispatch_max_retries`, exponential backoff). `InvalidTransition`, `ValueError`,
and `PreservedWorkerRun` re-raise immediately. The preserved-worker exception bypasses
message classification, even when its generated run ID contains `429`.
Other transient failures (OSError, timeout, 429/rate-limit,
exit 124/137/143 — see `_is_transient`) retry.[^preserved-dispatch-retry]

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
   `LOOP.md` via `_create_loop_run`, start the loop run, record `dispatched_at`.

On dispatch failure, cleanup releases the claim under its owner fence and discards
the worktree only after worker exit is confirmed. An unconfirmed stop preserves
the claim, running row, and worktree, then raises `PreservedWorkerRun`.
The owner fence prevents cleanup from releasing another run's claim.[^preserved-dispatch-retry]

[^preserved-dispatch-retry]: src/milknado/domains/execution/executor.py, `_should_retry_dispatch`, `_dispatch_once`, and `_cleanup_failed_dispatch`; tests/test_execution.py, `TestShouldRetryDispatch` and `test_dispatch_fence_loss_unconfirmed_stop_preserves_worktree`.

_Source: PR #510 CI failure and typed retry guard · Updated: 2026-10-01 · Supersedes: unconditional dispatch-failure cleanup description._

`_create_loop_run` renders the shared `dispatch.brief.render_brief` output, adds iteration-specific scaffolding, writes `LOOP.md`, starts the loop run, and returns its `run_id`.

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

- **Loop/MCP detached path** — the dispatching parent claims the node RUNNING
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

`NodeExecutionContext` replaces eight parallel per-node maps.
It keeps worker identity distinct from an adopted parent owner fence.
The record also holds the worktree, session, pinned target, base OID, review round, and configuration.
Pure review policy returns allow-merge, redispatch, or block decisions.
Executor retains worker calls, graph transactions, notifications, and Git effects.
Review audit precedes merge; audit failure or reviewer error blocks merge.
Notification failure remains separate from audit failure.
Review redispatch retains the worktree, session, owner fence, and capacity slot.
Terminal replay returns before merge, and unconfirmed cleanup preserves ownership and the worktree.[^deep-module-implementation]

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
| Orphan prune in `milknado_run_loop_start` (`mcp/loop.py`) | **degrade** — same; the run loop is never blocked |
| `WorktreeManager.discard` | unchanged — this IS the explicit destructive path |

`rebase_and_merge`'s landed check runs against `feature_branch` and the
worktree's HEAD *at removal time* — post-squash/rebase/ff — so the happy path
removes cleanly while a squashed-but-never-fast-forwarded HEAD (the old
silently-destroyed case) refuses. Non-refusal removal failures (nothing was
destroyed; the worktree is still on disk) stay warn-and-swallow so the node
lifecycle keeps moving.

## Shared loop lifecycle (`RunLoop`)

CLI scheduling and detached execution use `RunLoop` and `Executor`.
`RunLoop.run_node` drives one selected task without a display, sibling dispatch, or root completion.
Both drivers use shared completion, review redispatch, cancellation, timeout, and merge handling.
The detached module only composes adapters and records its parent run result.
An unconfirmed worker stop preserves node ownership and its worktree.
Detached supervision stays alive and retries the stop before finalizing the parent run.
The former `execution/headless.py` lifecycle twin is removed.

`ConcurrencyLimitReached` bypasses scheduler failure cleanup.
The CLI waits for external capacity; detached MCP starts return a structured `deferred` result.
Review redispatch retains node ownership and therefore retains one capacity slot.

`NodeClaimRejected` also bypasses failure cleanup: a losing scheduler must not fail another driver's task.
`PreservedWorkerRun` carries the worker identity and current owner fence through post-start failures.
The fence must update after `replace_run_id`; otherwise confirmed cleanup cannot release the reservation.
Shared completion treats an unrebased result as failure, not detached success.
The private `run_loop/_node.py` module supplies `NodeDriver` with an explicit `CompletionContext`.
It no longer shares state through a whole-RunLoop protocol or mixin.
Durable worker records now support identity-verified recovery after supervisor death.[^deep-module-implementation]

`RunLoop.state()` collects bounded immutable facts and one timestamp.
The pure projection computes presentation values without graph reads, process calls, or clock reads.
`Scheduler` owns active runs, attempts, progress, terminal history, counters, and local admission decisions.
Its private lock keeps snapshots coherent with concurrent completion and abandonment.
That lock contains only local state operations, never graph, process, or caller effects.
Graph claims remain authoritative; typed scheduler decisions only request effects.
The separate scheduling lock protects stop and dispatch admission.
Force-stop does not wait for that scheduling lock.[^deep-module-implementation]

## Controller authorization boundary

Normal local startup manages controller credentials without a manual export.
The graph keeps its existing credential hash and transactional, one-use review capabilities.
Controller registration loads or creates a credential under `$XDG_STATE_HOME/milknado/controllers`.
The POSIX default state root is `~/.local/state`.
Credential filenames use the registered hash, so moving a database does not change its credential lookup.[^controller-store]

Registration serializes through a SQLite write transaction.
It publishes the credential before committing a new hash.
The controller keeps generated credentials on its graph instance, not in the process environment.
Decision consumption does not automatically load the credential store.[^controller-store]

An existing registration accepts only its original credential.
A matching `MILKNADO_CONTROLLER_MASTER` imports that credential once for later starts.
Missing or mismatched credentials fail closed.
This path does not rewrite nodes, decisions, or authorization records.
There is no credential-free takeover or graph reset.[^controller-store]

A separate operator terminal uses `milknado graph review` with the same user state directory.
The command loads authorization and asks for confirmation.
A TTY selects the confirmation interface; it does not prove human presence.
Neither watch mode acquires a controller credential.
Attached watch retains its existing owner-fenced session controls, not goal-approval authority.[^controller-cli]

These checks are API guardrails, not process isolation.
Workers can submit proposals, but the MCP surface has no goal-approval tool.
Worker markers and secret filtering prevent ordinary worker calls from acquiring operator authority.
Same-user arbitrary code can remove markers, read user files, or modify SQLite.
Git worktrees and owner-only credential files do not prevent those actions.
Enforced isolation requires separate operating-system permissions or a sandbox outside this change.[^controller-workers]

### Windows storage and startup ordering

Windows uses `LOCALAPPDATA` as its default state root.
An absolute `XDG_STATE_HOME` overrides that root.
Storage creates missing ancestors and rejects reparse points through native handles.
The `milknado/controllers` directories and credential records require protected DACLs.
Their owner and sole full-access entry must match the process token's `TokenUser` SID.[^controller-windows]

Credential reads permit shared reads, but not shared writes or deletion.
Validation rejects empty records, records above 4096 bytes, and SHA-256 mismatches.
Publication flushes a new temporary file, then uses a non-replacing rename.
A publication race accepts only a destination with the expected hash and protection.[^controller-windows]

Asynchronous inline startup checks the protected branch before controller registration.
Registration precedes stale-run reclaim, node claims, worktree creation, and log creation.
Unauthorized startup must preserve the existing running node and its exact run identity.[^controller-order]

The Windows workflow runs native storage, startup, and merge-lock tests.
A macOS test pass does not establish native Windows behavior.
Keep the controller pull request in draft until that workflow passes.[^controller-windows-ci]

[^controller-windows]: src/milknado/domains/graph/_windows_controller_storage.py:34-105,123-174,206-269.
[^controller-order]: src/milknado/app/run.py:538-584; tests/test_run_inline_worktree.py.
[^controller-windows-ci]: .github/workflows/windows-controller.yml:13-28; approved controller draft publication requirement.

[^controller-store]: src/milknado/domains/graph/controller_capability.py; src/milknado/domains/graph/graph.py, register_controller_master and decide_goal_review.
[^controller-cli]: src/milknado/cli/graph.py, review; src/milknado/cli/run.py, watch.
[^controller-workers]: src/milknado/domains/dispatch/runner.py, build_worker_env; src/milknado/loop/_agent.py, _build_spawn_env; src/milknado/loop/sessions/_process.py, start_process; src/milknado/mcp/goal_review.py.

_Source: controller startup and Windows storage source; selected Cure regressions · Updated: 2026-09-21 · Supersedes: platform-neutral default-root claim; retains the API-guardrail boundary._

## Subprocess workers & run-state (runner.py, _runstate.py)

`run_headless` (blocking, via `_execute`) and `start_headless_async` (detached
thread, via `_execute_cancellable`) spawn a worker through `_spawn_worker`:
brief piped to stdin, combined stdout/stderr to a log file, cwd = worktree.

- **Worker allowlist** — `_validate_worker_argv` checks the *basename* of argv[0]
  against `{claude, codex, cursor-agent, gemini}`, defeating both prefix tricks
  (`claude-evil`) and absolute paths. Guards the explicit MCP arg, the
  `$MILKNADO_WORKER_CMD` env, and the default `claude -p`.
- **Env scrubbing** — `build_worker_env` filters the inherited environment.
  Dispatch and native agent launchers remove `MILKNADO_CONTROLLER_MASTER` after overrides.
  They set `MILKNADO_WORKER_CONTEXT=1` to reject controller acquisition and goal approval through supported worker paths.[^controller-workers]

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



### Loop pipe ownership

A reader thread owns each `Popen` stdout or stderr stream until EOF.[^pipe-ownership]
Cleanup must not call `os.close(stream.fileno())` while the Python stream remains open.
A detached descendant can inherit a pipe and keep its reader alive after the direct child exits.
Cleanup therefore bounds the join and leaves each live reader with its stream.
The reader closes the stream after EOF or a terminal read error.
Raw descriptor closure leaves a stale `TextIOWrapper`.
Later finalization can close an unrelated descriptor after the operating system reuses the descriptor number.

[^pipe-ownership]: `src/milknado/loop/_agent.py:878-899,927-976`; `tests/loop/test_agent.py:1835-1884`

## Native worker sessions

Native worker sessions separate decoded protocol events from raw diagnostic output.
`LoopSessionMixin` selects supported worker commands and attaches durable session storage
(`src/milknado/adapters/_loop_session.py:23`).

- OMP uses RPC, Claude uses bidirectional stream-json, and Codex uses app-server messages.
- `SessionChannel` persists input admission before it accepts a command.
- `queued` means Milknado accepts the input. `submitted` means the channel releases it to the protocol.
- `delivered` requires a vendor receipt. A queued or submitted input does not prove delivery.
- Event IDs include an invocation prefix. Reused vendor IDs cannot overwrite an earlier iteration's transcript.
- Permission decisions become approved or denied only after their command write succeeds.
- Shutdown rejects unsent input. It marks submitted input without a receipt as unconfirmed and cancels pending permissions.

Native completion uses decoded result text, never raw stdout. A replayed prompt can contain the completion tag.
The engine trusts the native completion flag and still runs its completion verifier
(`src/milknado/loop/sessions/_runtime.py:215`; `src/milknado/loop/engine.py:235`).
The verifier requires a committed or stageable change and the configured quality gates.
A completion promise alone does not complete a task (`src/milknado/domains/execution/completion.py:22`).

The runtime bounds protocol frames and retained output. On POSIX, it retains the process group after the leader exits.
Normal completion, timeout, and force stop use the shared lifecycle boundary.
Cleanup targets owned groups and identity-verified observed descendants, subject to the limits below.[^deep-module-implementation]

Generic and native worker paths do not have identical Windows containment.
Generic workers use a kill-on-close Windows Job Object; native sessions currently use direct process termination.
A shared lifecycle refactor must preserve these platform differences rather than assume equivalent descendant cleanup.[^worker-process-boundaries]

The P0 implementation now enforces design decision F-8, accepted on 2026-09-29.
It provides bounded cleanup of owned groups and identity-verified observed descendants, not strict containment.
Unobserved descendants that detach between snapshots can escape cleanup; the user explicitly accepts that limit.
Known unconfirmed targets still prevent ownership release and a confirmed-cleanup result.
F-7 now selects one cleanup lifeline per active worker invocation, not a shared guardian.
This adds one helper per worker but isolates cleanup ownership and failures.
Supervisor, lifeline, and actual worker identities remain distinct; lifeline exit cannot prove worker exit.
F-9 refuses ambiguous kills when a recorded leader is dead and surviving group ownership cannot be verified.
Cleanup stops verified targets only; unresolved identity preserves node ownership and produces a recovery diagnostic.
Historical group numbers and recycled leader identities do not prove cleanup completed.
F-5 closes admission when shutdown is recorded and includes launches already in progress in bounded cleanup.
An in-flight launch may create a process after signal arrival; it remains tracked under the same cleanup deadline.
Handlers must not acquire application locks; normal control flow performs cleanup.
F-10 selects replacement when a lifeline dies unexpectedly while its supervisor survives.
The implementation keeps worker parentage, protocol streams, and exit collection in the supervisor runtime.
Replacement transfers monitoring and cleanup responsibility, not the worker's operating-system parentage.
The replacement must retain worker identity and descendant evidence and reject stale helper acknowledgments.
F-11 accepts the two-failure limit: automatic cleanup is not guaranteed if the supervisor dies before a failed lifeline's replacement becomes ready.
Durable records remain for later recovery; confirmed replacement readiness restores protection for the current invocation.
This design adds no independent backup cleanup owner.
F-12 stops the worker when lifeline replacement keeps failing; replacement does not continue indefinitely.
The supervisor performs verified cleanup without depending on the failed lifeline and escalates to SIGKILL when necessary.
Confirmed cleanup fails only the affected run; unconfirmed exit preserves ownership and recovery records.
The shared private lifecycle boundary owns launch protection, lifeline replacement, and cleanup.
Durable ownership covers graph-backed, reviewer, verifier, standalone, and raw-manager loop workers without synthetic graph nodes.
An exec gate releases user work only after durable worker identity and initial lifeline readiness.
READY binds the durable snapshot sequence after EOF monitoring is armed.
Observation-in-progress markers preserve unresolved coverage when discovery or persistence fails.
A replacement can launch at most three times per invocation, within an eight-second episode.
Successful replacement does not reset that attempt count or the first shutdown deadline.
Real-process, SQLite, and Git tests validate these boundaries; Windows parity remains unverified.
The scheduler, projection, review policy, and node-context refactors preserve this lifecycle boundary.[^bounded-orphan-cleanup][^deep-module-implementation]

PR #503 review adds these implementation constraints:
Node-scoped recovery stops workers only when their durable supervisor identity is gone.
Live, unknown, or mismatched supervisors remain untouched and make node recovery incomplete.
Unassociated recovery still skips live supervisors; explicit run cancellation retains its stop policy.[^pr503-recovery]
Cancellation reserves time for SIGKILL confirmation within the original timeout.
Preserved-stop retries remain inside signal supervision on normal and exception paths.[^pr503-signals]
Protection failure remains distinct from ownership confirmation.
Production completion fails after replacement exhaustion even when the worker handles SIGTERM with exit status zero.
Confirmed cleanup still closes durable evidence and releases its admission ticket.
Unconfirmed cleanup retains ownership and reports protection failure alongside the cleanup diagnostic.
Helper-reap errors retain their cause; reader draining remains bounded before cleanup reports failure.
Shutdown reaps the helper after monitor quiescence and EOF; helper exit never proves worker exit.
Generic runners close capture and log sinks even when cleanup raises.[^pr503-lifecycle]
A child that disappears during process-group sampling is absent, not uncertain.
Other observation errors retain unresolved evidence.[^pr503-observation]
The worker-store FileLock explicitly uses mode 0600.
The supported minimum filelock 3.19.1 otherwise changes a precreated private lock to 0644 and breaks the next store open.[^pr503-filelock]
Shutdown fixture markers use atomic replacement so visibility implies complete identity data.[^pr503-marker]

[^pr503-recovery]: src/milknado/domains/dispatch/reap.py:117-139; tests/test_worker_recovery_errors.py.
[^pr503-signals]: src/milknado/adapters/process.py:100-122; src/milknado/mcp/_loop_node_runner.py:135-152; tests/test_process_termination_budget.py; tests/test_loop_node_runner_signal_subprocess.py. The signal fixture exercises the real runner and OS signals with a stub RunLoop.
[^pr503-lifecycle]: src/milknado/loop/_process_lifecycle.py:182-296; src/milknado/loop/_agent.py:833-914,1082-1143; tests/loop/test_lifecycle_acceptance_exhaustion.py; tests/loop/test_lifecycle_error_paths.py; tests/loop/test_all_worker_context.py.
[^pr503-observation]: src/milknado/loop/_process_identity.py:43-49; tests/loop/test_process_identity_errors.py.
[^pr503-filelock]: src/milknado/domains/graph/worker_evidence.py:93; tests/test_worker_evidence_errors.py. Repeated-open regression fails before the fix and passes with filelock 3.19.1.
[^pr503-marker]: tests/test_shutdown_subprocess.py:50-54.

_Source: PR #503 source review and focused regression tests · Updated: 2026-09-30 · Supersedes: the 2026-09-30 blanket implementation-verification claim; F-12 requires production failure propagation._


[^bounded-orphan-cleanup]: Durable spec `reap-orphaned-loop-workers.md`, Decisions F-5/F-7–F-12, Acceptance AC-6/AC-9/AC-12–AC-18; user selections in the 2026-09-29 design dialogue.

[^native-process-containment]: src/milknado/loop/sessions/_process.py:171-202,255-286
[^worker-process-boundaries]: src/milknado/loop/_agent.py:190-235,786-800; src/milknado/loop/sessions/_process.py:171-192,255-286



### Synthetic worker interpreter

Run Python worker fixtures with the test interpreter, not an interpreter selected through `env python3`.
The lifecycle fixture has a two-second execution deadline.
On 2026-09-13, the environment selects `/opt/homebrew/bin/python3` and delays the first frame beyond that deadline.
Five fresh-directory measurements range from 0.583 to 3.288 seconds without the session runtime.
The current virtual-environment interpreter takes 0.020 to 0.028 seconds for the same script.
An in-memory shebang correction makes all ten lifecycle tests pass without changing deadlines or runtime code.[^fixture-interpreter]
This explains the reproduced no-frame startup timeout; it does not prove that every historical lifecycle failure has the same cause.

[^fixture-interpreter]: `tests/test_session_lifecycle.py`, `_SCRIPT`, `_Scenario`, and `_worker`; coordinator diagnostic sessions `77586`, `91636`, and `58094` on 2026-09-13 UTC. Session `91636` measures the same fake tool-cap worker in fresh temporary directories with real pipes and process groups.



Later measurements show that the interpreter-header correction is insufficient on this host.
The first direct execution of one generated script takes 2.113 seconds; repeated executions take 0.032–0.040 seconds.
Explicit interpreter execution takes about 0.02–0.03 seconds for the same script.
Neither sandbox removal nor a resolved interpreter path removes the direct-execution delay.
The host mechanism remains unproven; these measurements do not identify a production runtime defect.[^fixture-cold-launch]

A fixture-only experiment uses a `claude` symlink to the current interpreter and passes `claude.py` as its first argument.
All ten lifecycle tests pass with unchanged deadlines and assertions.
Review confirms that this preserves Claude adapter selection, protocol flags, argument positions, and real process behavior.
The fixture no longer tests shebang resolution, which is outside the lifecycle contract.
This experiment remains uncommitted and does not prove a green full gate.[^fixture-explicit-launch]

[^fixture-cold-launch]: Parent diagnostic sessions `14490`, `25288`, `67378`, and `51691` on 2026-09-13.
[^fixture-explicit-launch]: Parent in-memory diagnostic `70356`: 10 passed in 5.56 seconds; read-only review by `task38_recovery_review` confirms `cmd[0]` basename selection and appended Claude flags.



The explicit-interpreter fixture correction now exists in the preserved task #38 worktree.
The lifecycle, failure-path, and runtime suites pass all 24 tests in 7.54 seconds.
Severity review approves the implemented callers, including the reader subprocess and engine command.
Fresh taste-test round 1 passes all seven lenses.
The full gate remains pending behind the selection-writer freeze barrier.[^fixture-implemented]

[^fixture-implemented]: `task38_explicit_fixture_cure`, `task38_recovery_review`, and `task38_explicit_fixture_taste` handbacks on 2026-09-13; `tests/test_session_lifecycle.py`, `tests/test_session_failure_paths.py`, and `tests/test_session_runtime.py` in `milknado-38-tree-left-shared-workspace-and-2`.



### Coordinator action and reservation guards

Coordinator commands require an exact durable coordinator, provider-family, and provider-session binding before reserving a receipt.[^pr518-action-binding]
A display or discovery link does not authorize provider input.
The domain submission port uses identity properties and one submission method.
The existing `RuntimeSession` implements that port without a separate wrapper or unused factory.[^pr518-live-port]
Its method delegates to native admission and incarnation fencing.
Uncertain submission keeps its durable receipt; retries do not submit the action again.[^pr518-action-replay]

Reserved launches recheck prerequisites inside the transaction before claiming.
Already-running retries retain their idempotent branch.[^pr518-reserved-launch]
Launch failure retains reservation ownership while any same-node worker has no confirmed end record.
Confirmed cleanup permits the normal failure path.[^pr518-reservation-cleanup]
These checks preserve durable-before-submit ordering and ownership instead of relaxing them.

[^pr518-action-binding]: src/milknado/domains/coordinator/commands.py:135-150; tests/coordinator/test_action_binding_guards.py:18-42.
[^pr518-live-port]: src/milknado/domains/coordinator/commands.py:17-24,151-159; src/milknado/loop/sessions/_lifecycle.py:35-50.
[^pr518-action-replay]: tests/coordinator/test_action_receipts.py:42-135.
[^pr518-reserved-launch]: src/milknado/domains/graph/_execution_groups.py:191-205; tests/coordinator/test_reservation_guards.py:14-82.
[^pr518-reservation-cleanup]: src/milknado/domains/graph/_group_reservation.py:89-118; tests/coordinator/test_reservation_guards.py:85-116.



_Source: approved PR 518 guard and submission-port corrections · Updated: 2026-10-09._

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
deposit soft-no-ops instead of raising "run not found". The CLI run loop
(`milknado run`) now injects the same three variables — `MILKNADO_NODE_ID`,
`MILKNADO_RUN_ID`, `MILKNADO_PROJECT_ROOT` — through `RunConfig.env`, set by
`Executor._create_loop_run` and merged into every agent spawn of the run.
The same per-node flavor replace in `RunLoop._dispatch_batch` also carries
`max_iterations`, `attempt_timeout_seconds`, and `completion_timeout_seconds`
(attempt timeout × max iterations) from the flavor profile, matching the MCP
node runner, so a CLI-dispatched attempt is bounded.

Progress messaging and coordinator→worker inbound are deliberately NOT built;
the `run_messages` shape permits them later (YAGNI — spec non-goal).

## MCP run surface — five tools, one schema (#82, #83, #107)

Five coordinator-facing MCP tools drive runs: `milknado_run_inline` (blocking),
`milknado_run_inline_start` / `milknado_run_inline_poll` (async in-process worker),
and `milknado_run_loop_start` / `milknado_run_loop_poll` (detached
worktree-isolated loop). All five return the **unified superset schema**
`RunDict` (`mcp/_core.py`): `run_id, node_id, status, exit_code, timed_out,
rebased, log_path, summary`, every field nullable where it doesn't apply
(`summary` is None until a poll tails the log; `rebased` is None for non-loop
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
  then forks on run type. A **detached-loop** run has its own process group, so
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

## Worker-dispatch families — single-shot (3) vs loop (4), and why both exist

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
- **Family 4 = iterate-until-gates loop** — `milknado_run_loop_start` /
  `_poll` (`mcp/loop.py:62` / `:89`). Detached
  `Popen([... _loop_node_runner ...], start_new_session=True)`, no stdin, runs
  in its **own git worktree+branch**, loops until `quality_gates` pass through
  `RunLoop.run_node`, then rebase-merges back. It refuses to start without quality gates.
  It survives an MCP server restart; polling is read-only.
  CLI and detached execution share the loop lifecycle described above.

**Why Family 3 exists (rationale the code doesn't state):** to dispatch a node
to a **different harness than the one orchestrating** and **block on the
single-shot result without polling**. The coordinator (say Claude Code)
overrides `worker_cmd` to point at another agent — e.g. a local-model-backed CLI
for cheap or offline nodes — and `milknado_run_inline` blocks until that foreign
worker exits, returning the result inline. It has worktree isolation by default, but no gate loop or polling cycle. `worker_cmd` is the
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
detached runner does. Worktree isolation lets many loops run in parallel
without trampling each other; detachment lets the loop outlive the MCP server.

**Coordinator vs worker — who may call these (verified against the allowlist):**
both run-dispatch families are **coordinator-facing**; *neither* is granted to
spawned workers. `WORKER_ALLOWED_TOOLS["claude"]` (`agent_argv.py`) gives a
worker only two milknado MCP tools — `milknado_track_follow_up` and
`milknado_deposit_result` — never the run-dispatch tools. Family 4 additionally
carries a **hard, permanent prohibition**: *"COORDINATOR-ONLY: never add these
to WORKER_ALLOWED_TOOLS"* (`mcp/loop.py:1-12`), because a worker that could
start sub-loops would recursively fork worktrees. (A casual read of the
code can mis-state Family 3 as worker-allowed — it is not; re-check
`WORKER_ALLOWED_TOOLS` before asserting otherwise.)

| | Family 3 — single-shot | Family 4 — loop |
|---|---|---|
| Tools | `milknado_run_inline` / `_start` / `_poll` | `milknado_run_loop_start` / `_poll` |
| Process | blocking caller, or in-process daemon thread | detached subprocess (`start_new_session=True`) |
| Brief delivery | piped on stdin | node/run id via env + CLI args (no stdin) |
| Iterations | one pass | loop until quality_gates pass / timeout |
| Working tree | own worktree+branch by default (`THIS_BRANCH` opts into shared) | own worktree+branch, rebase-merge on success |
| Quality gates | none | required (refuses if unset) |
| Survives server restart | no | yes (pid in SQLite) |
| Loop / retries owned by | caller | shared `RunLoop` lifecycle |
| Granted to workers? | no (coordinator-facing) | no — explicit permanent prohibition |
| Reach for it when | block on a (possibly foreign-harness) single shot, no poll | hand off the whole gated loop, walk away |

## Orphan recovery & reconciliation

A worker can vanish (server crash) leaving a `running` run row and a RUNNING
node. `reconcile_orphan_node` is the shared three-call recovery:

1. `fail_stale_running_runs` — flip `running` rows older than
   `timeout + _STALE_GRACE_SECONDS` (30s) to `failed` — **skipping (and
   logging) a row whose recorded pid is still alive** (#54), so a slow live
   worker is never force-failed by the sweep.
2. `fail_stale_running_runs` has marked stale rows failed, then
   `latest_terminal_run(node_id, run_id)` selects the terminal row for the
   still-owned fence. The producer requires `run_id` in that query, so a stale
   later-ended row cannot mask the current owner's row.
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

## Shared brief and LOOP.md

`brief.render_brief` is the shared node-context markdown for native/MCP and
loop workers: goal context, completed prerequisites, owned files, specs, and
instructions. Loop dispatch embeds that exact brief in `LOOP.md` and adds only
the iteration-specific findings, gates, follow-up protocol, and completion
sentinel.

## Key files

- `src/milknado/domains/execution/executor.py` — `Executor`, `WorktreeManager`, dispatch/complete state machine, fencing.
- `src/milknado/domains/execution/run_loop/__init__.py` — `RunLoop` orchestration, fact collection, and effects.
- `src/milknado/domains/execution/run_loop/_scheduler.py` — synchronized local state and typed dispatch decisions.
- `src/milknado/domains/execution/run_loop/_projection.py` and `state.py` — pure projection and immutable facts.
- `src/milknado/domains/execution/_node_context.py` and `_review_policy.py` — per-node facts and pure review decisions.
- `src/milknado/loop/_process_lifecycle.py` — protected worker lifetime, helper replacement, and bounded cleanup.
- `src/milknado/domains/execution/run_loop/_node.py` — bounded `run_node` driver and shared terminal, timeout, and stop handling.
- `src/milknado/domains/execution/_models.py` — execution results and typed dispatch failures.
- `src/milknado/domains/dispatch/brief.py` — `render_brief` (shared worker context and result-deposit instructions).
- `src/milknado/adapters/loop.py` — LOOP.md loop scaffolding.
- `src/milknado/domains/dispatch/runner.py` — subprocess spawn, async worker, cancel, orphan recovery, `reconcile_node_status`.
- `src/milknado/domains/dispatch/_runstate.py` — run-id format, log tail, cancel sentinel (run *state* lives in the SQLite `runs` table).
- `src/milknado/adapters/tmux.py` — `TmuxAdapter`, `RunWindow`, exact-match targeting, the POSIX-sh window wrapper.
- `src/milknado/domains/dispatch/tmux_run.py` — `ensure_tmux_ready` (fail-closed), `execute_in_window` (run-inline pane waiter), `cleanup_run_window` / `reconcile_run_window` (per-row lifecycle), `resolve_attach_target`.
- `src/milknado/domains/common/agent_argv.py` — `WORKER_ALLOWED_TOOLS` (per-vendor worker tool allowlist), `_ALLOWED_WORKER_EXECUTABLES` / `validate_worker_argv` (worker-cmd basename gate), `resolve_*_agent_command`.
- `src/milknado/domains/graph/_persistence.py` — `runs` / `run_messages` repo (`start_run`, `finish_run`, `deposit_run_message`) and the `goal_claims` table schema.
- `src/milknado/domains/graph/_goal_claims.py` — goal-claim repo + fencing helpers (`claim_goal_row`, `release_goal_row`, `ancestor_goal_claimed_by_other`, `claim_or_reclaim_goal`); `MikadoGraph.claim_ancestor_goal_for_dispatch` (`graph.py`) is the dispatch-time entry point.
- `src/milknado/mcp/run.py` — `milknado_run_inline*` (Family 3), `milknado_run_list`, `milknado_run_cancel`, `milknado_deposit_result`.
- `src/milknado/mcp/loop.py` — `milknado_run_loop_start` / `_poll` (Family 4, COORDINATOR-ONLY).
- `src/milknado/mcp/_core.py` — `RunDict` unified run-result schema, the shared `FastMCP` instance, and the `resolve_project_root` / `open_graph` / status-kind-flavor parsers. (The goal-claim fencing this module once held now lives on `MikadoGraph.claim_ancestor_goal_for_dispatch`; see `domains/graph/`.)

[^deep-module-implementation]: P0 `ad0110b`, P1 `cae3a54`, P2 `568c45e`, P3 `95862f8`; `loop/_process_lifecycle.py:85-198`, `execution/run_loop/_scheduler.py:73-165`, `execution/run_loop/_projection.py:64-78`, and `execution/_node_context.py:32-42`. Per-unit `just check-llm` gates pass. Runtime coverage includes `tests/loop/test_lifecycle_acceptance.py`, `tests/test_orphan_worker_recovery.py`, `tests/test_run_loop_scheduler.py`, and `tests/test_adversarial_review_runtime.py`.

_Source: PR #488 and the verified deep-module commits cited above · Updated: 2026-09-30 · Supersedes: pending orphan-worker implementation, absent durable recovery, shared driver mixin, and combined scheduling/presentation ownership. The accepted F-5/F-7–F-12 limits remain._
