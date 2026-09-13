# Review verdicts and durable audit records

## Historical failures

The August–September 2026 campaign exposes two distinct review failures.
Unparseable reviewer output becomes a code rejection and consumes another worker iteration.
A second execution cycle can reuse a node's review round and collide with the audit primary key.
The earlier executor also records rejections but not approvals.
Thus, an absent approval row does not prove that historical review never runs.

The meta run reproduces the first failure with a progress-only Opus response.
Node 3 receives no code correction, but the old parser starts another worker iteration.
See [meta verification](../conventions/meta-verification.md) for that run's evidence.

## Repair contract

The reviewer must emit exactly one valid verdict tag.
Missing, conflicting, duplicate, malformed, and unclosed verdict markers produce a reviewer error.
The parser retains invalid output as findings instead of inventing code corrections.
The domain review-result protocol carries the error state explicitly.

Reviewer errors block completion under both rejection policies.
They preserve the worktree and return control without another worker run.
They do not consume the code-revision round budget.
A valid rejection still follows the configured retry and rejection policy.

Every review result writes a durable `node_reviews` record before merge.
Approval, rejection, and reviewer error remain distinct audit verdicts.
A required audit failure blocks merge, including the warn-policy path after exhausted rejection rounds.
The run-loop display and log identify audit failures.
Worker notification remains separate from the required audit write.

An atomic SQLite insert allocates the next audit sequence for each node.
The sequence uses stored records, not the executor's in-memory revision budget.
This distinction prevents duplicate round keys after restart or concurrent writers.
It adds no migration backfill or compatibility layer.



A findings-file write failure must not terminate the whole run loop.
The recovery path logs file errors and still records a database error verdict.
It blocks that node without merge or worker retry.
The regression exercises real completion with an injected filesystem write error.



The graph run sub-facade owns audit reads through `graph.runs.reviews_for_node`.[^1]
Tests and cross-slice code must not query `node_reviews` through the private graph connection.
That bypass can hide a missing public reader while the storage assertions still pass.

[^1]: src/milknado/domains/graph/_facades.py:101-105; src/milknado/domains/graph/_run_persistence.py:260-276; tests/test_adversarial_review_runtime.py:406-424

## Regression evidence

`tests/test_adversarial_review_runtime.py` checks approval persistence, audit failures, malformed output, and preserved-worktree handbacks.
`tests/test_graph_persistence.py::test_review_sequence_is_atomic_across_connections` checks concurrent database connections.
The completion-handler tests check visible audit-failure reports.
The combined repository gate remains `just check-llm`.

## Recovery boundary

Read the stored findings before deciding whether code needs a correction.
Repair the reviewer configuration or audit storage when the failure belongs to that system.
Resume through the supported dispatch path after the cause is fixed.
Do not mark a node done with direct SQL to bypass review or completion checks.

A blocked node is a scheduler hold, not an enforced human-approval lock.
Direct dispatch can claim it.
The repair does not change that authorization model.



## Authorized retirement of obsolete TUI records

The user authorizes deletion of nodes 14, 15, and 25 after evidence preservation on 2026-09-12.
This retirement does not mark those nodes completed or bypass their completion checks.
It does not approve the historical rejected review or declare the new steering feature complete.

| Node | State before retirement | Preserved evidence |
| --- | --- | --- |
| 14 | blocked | Runtime PR [441](https://github.com/paulnsorensen/milknado/pull/441) merges at 2026-09-10T08:46:52Z, commit `352e2e7ded0bd805c63d7cd080d848c7186b438e`. |
| 15 | pending, prerequisite 14 | Session/Changes/Details PR [442](https://github.com/paulnsorensen/milknado/pull/442) merges at 2026-09-10T09:14:25Z, commit `7dee80f78113b27b95f9a88459e5675716070ad1`. |
| 25 | running node label; terminal run | Run `node-25-20260910T040905Z-88828f00` reports done, exit 0, and a historical reject verdict. |

The GitHub CLI verifies both merge records before deletion.
Node 14 records a missing immutable dispatch base; node 15 records that prerequisite as its completion barrier.
Node 25's zero process exit means the reviewer finishes, not that it approves the code.
The full historical review follows; its file references describe that earlier checkout, not current correctness.
The original checkout is `/Users/paul/Dev/milknado/.worktrees/structured-tui-sessions`.
Deletion targets these three records only, with `cascade=false`; worktrees and published PRs remain untouched.

<details>
<summary>Preserved node 25 review</summary>



Non-cascading deletion initially stops because the graph treats prerequisite edges as children.
The recorded edges are 15→14, 14→33, 25→22, and 25→23.
Detach only those incident edges through the public graph API before deleting the authorized records.
Retain completed records 22, 23, and 33 and their archived history.



Retirement completes: each non-cascading delete returns `deleted: 1`.
A graph query including archived records confirms IDs 14, 15, and 25 are absent.
The same query confirms completed records 22, 23, and 33 remain present.

# Runtime Review

**Verdict: REQUEST CHANGES (`reject`)**

## Blocker

None.

## High

- **[H1 — merge blocker] Terminal permission receipts do not release pending-input capacity.**  
  `SessionChannel.submit()` counts every accepted command in `_pending` at `src/milknado/loop/sessions/_channel.py:108-121`. The terminal permission branch only removes `_permissions` at `src/milknado/loop/sessions/_channel.py:276-280`. `_acknowledge()` runs only for terminal `user` events at `src/milknado/loop/sessions/_channel.py:281-292`. Claude publishes `approved` or `denied` after a successful transport write at `src/milknado/loop/sessions/_claude.py:152-159`. Codex publishes the same terminal states at `src/milknado/loop/sessions/_codex_approval.py:70-83`. Neither event clears the submitted command.  
  **Failure path:** Each successful Claude or Codex permission decision permanently consumes one `max_inputs` slot. After 64 decisions, the default channel rejects all later input. Normal shutdown also emits a false `user:unconfirmed` receipt for the completed decision at `src/milknado/loop/sessions/_channel.py:163-171`.  
  **Fix:** Correlate terminal permission decisions with pending tokens. Acknowledge the token only after the terminal permission event persists. Preserve current shutdown behavior for decisions that never reach a terminal state. Add approved and denied capacity regressions.  
  **Confidence:** Certain.

- **[H2 — merge blocker] Repeated cleanup can signal a recycled, unrelated process group.**  
  The retained PGID is stored at `src/milknado/loop/sessions/_process.py:34,265`. Every `terminate()` call retrieves it at `src/milknado/loop/sessions/_process.py:152-159`. No path removes it. The runtime calls `terminate()` after leader exit at `src/milknado/loop/sessions/_runtime.py:179-180`. `finish_process()` calls it again at `src/milknado/loop/sessions/_process.py:181-190`. Final cleanup calls it again at `src/milknado/loop/sessions/_process.py:201-208`. The `pgid == proc.pid` check proves numeric equality only.  
  **Failure path:** One call kills and reaps the owned group. The OS reuses that PID as another group’s PGID. A later cleanup call sends SIGTERM or SIGKILL to the unrelated group at `src/milknado/loop/sessions/_process.py:161-165`.  
  **Fix:** Consume or clear retained group identity after group termination completes. Make later cleanup idempotent without another group signal. Keep the spawn-time identity guard.  
  **Confidence:** Certain.

## Medium

- **[M1 — merge blocker] Protocol and stderr limits count decoded characters instead of bytes.**  
  `_read_frame()` uses `TextIO.readline()` and checks `len(str)` at `src/milknado/loop/sessions/_process.py:70-81`. Both pipes use `text=True`, so decoding occurs before the check at `src/milknado/loop/sessions/_process.py:246-260`.  
  **Failure path:** A frame containing 1,048,576 four-byte UTF-8 characters passes `MAX_FRAME_SIZE` while consuming about 4 MiB. The approved design requires a strict pre-decode byte limit. The 64 KiB stderr limit has the same expansion. Invalid bytes are also replaced before framing.  
  **Fix:** Read bounded binary chunks. Enforce byte limits before decoding. Decode only complete bounded frames with the intended UTF-8 error policy.  
  **Confidence:** Certain.

- **[M2 — merge blocker] Structured sessions omit stderr from the public output callback.**  
  `OutputLineCallback` promises raw lines with either stream name at `src/milknado/loop/_agent.py:67-68`. `StreamContext` exposes only `on_stdout` at `src/milknado/loop/sessions/_stream.py:14-21`. `consume()` publishes stderr as a session error and returns at `src/milknado/loop/sessions/_stream.py:36-45`. Only stdout reaches `spec.on_output_line()` at `src/milknado/loop/sessions/_runtime.py:132-135`.  
  **Failure path:** A caller that sets cancellation or completion state from stderr never receives that line. The worker continues until another stop condition or timeout.  
  **Fix:** Forward every consumed raw line through the callback with its exact stream literal. Continue decoding protocol frames only from stdout. Add a stderr callback regression at the runtime seam.  
  **Confidence:** Certain.

## Low

No optional cleanup findings.

## Checked constraints

- `SessionProtocol.actions` remains read-only at `src/milknado/loop/sessions/_protocol.py:22-27`.
- Codex partial mixins retain `ABCMeta`.
- Final Codex composition preserves command serialization and completion ownership.
- Claude preserves `ProtocolStep.after_write_events` while combining frames at `src/milknado/loop/sessions/_claude.py:167-190`.
- Runtime publishes after-write events only after `write_commands()` succeeds at `src/milknado/loop/sessions/_runtime.py:126-130`.
- SQLite row-helper callers use graph connections configured with `sqlite3.Row`. I found no current runtime defect there.
- Pending queued and submitted shutdown paths retain their prior behavior. H1 affects completed permission cleanup.
- No files were modified.
- No tests, builds, linters, formatters, or quality gates were run.

## Merge recommendation

**Do not merge.** Fix H1, H2, M1, and M2. Then run the main-owned `just check-llm` gate and focused runtime regressions.

The complete report and `reject` verdict were deposited for run `node-25-20260910T040905Z-88828f00`.


</details>

Evidence sources: GitHub `gh pr view` for 441 and 442; `milknado_get_node` for each retired ID; exact-run poll.
The historical report source is `.milknado/runs/node-25-20260910T040905Z-88828f00.log`.

## Review checks retained from the historical campaign

Reject broad type-check suppressions that hide unrelated errors.
Check that tests and assertions remain in place.
Check that concurrency tests retain their barriers.
Verify claimed fixes in the diff and executable tests.
Do not accept an agent's summary as the only evidence.
