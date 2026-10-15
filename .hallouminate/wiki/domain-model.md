# Milknado domain model

The domain model distinguishes logical task attempts, run-row evidence, native task reservations, and execution ownership.
Issue #495 implements these execution-context definitions; existing ownership terms remain unchanged.

## Execution ownership

These lifecycle distinctions retain the selected design responsibilities and now map to the implemented runtime.[^design][^implementation]

**Supervisor** — The host runtime owns the run and the actual worker handles.
_Avoid_: using worker identity as supervisor identity.
_Code_: `src/milknado/domains/common/process.py:21-27`; `src/milknado/loop/_process_lifecycle.py:27-49`.

**Worker invocation** — One actual agent process and its execution identity. Replacing a lifeline does not create another invocation.
_Avoid_: using a lifeline PID as the worker PID.
_Code_: `src/milknado/domains/common/process.py:13-18`; `src/milknado/loop/_process_lifecycle.py:85-138`.

**Lifeline** — A per-worker monitor for supervisor liveness and verified cleanup. The supervisor retains agent streams and exit collection.
_Avoid_: stream proxy, shared guardian.
_Code_: `src/milknado/loop/_lifeline.py:78-124`; `src/milknado/domains/common/process.py:30-35`.

**Covered targets** — Owned groups and identity-verified observed descendants included in bounded cleanup.
_Avoid_: every possible descendant.
_Code_: `src/milknado/loop/_process_lifecycle.py:202-238`; F-8/F-9.

**Confirmed cleanup** — Exit confirmation for every covered target. Signal delivery, leader exit, or closed stdout does not establish it.
_Code_: `src/milknado/loop/_process_lifecycle.py:222-234`; `src/milknado/loop/_lifeline.py:119-124`.

See [execution architecture](./architecture/execution.md) for failure policies, verification evidence, and accepted containment limits.

[^design]: Durable spec `reap-orphaned-loop-workers.md`, Decisions F-7–F-12 and Acceptance AC-3/AC-13–AC-18.
[^implementation]: P0 implementation through `ad0110b`; current source `7e06824`. Real-process and durable-record tests pass with the complete project gate.

_Source: Approved lifecycle design and verified P0 implementation · Updated: 2026-09-30 · Supersedes: proposed-only status and pre-implementation code references; canonical terms remain unchanged._

## Bounded task attempts — logical executions and run rows

A logical task attempt is one task execution; a run row records evidence, and a native task attempt identifies its reservation.
The graph ledger implements these issue #495 definitions across the execution routes.[^attempt-domain-design]
The [execution ADRs](./architecture/execution.md#bounded-node-attempts) record the choices and their consequences.

### Logical task attempt and run row

**Logical task attempt** — One task execution across its linked outer, worker, and review run rows.
One confirmed task failure consumes one attempt.
Cancellation and preserved merge outcomes consume no failed attempt; an unmerged successful execution retains recovery pins.[^attempt-domain-repair]
The graph-owned `task_attempts` ledger is the approved new accounting entity.
_Avoid_: failed run row, vendor iteration.[^attempt-domain-f2]

**Run row** — Persisted execution evidence; multiple rows can belong to one logical task attempt.
After confirmed cleanup, a stale linked row can close without recounting its settled attempt or changing the node disposition.[^attempt-domain-repair]
_Avoid_: treating every FAILED row as another failed task execution.
_Code_: `RunRecord`, src/milknado/domains/graph/_run_persistence.py:65-108.[^attempt-domain-source]

### Native task attempt and current owner

**Native task attempt** — Existing reservation identity for a native group task execution.
Its `TaskAttempt.attempt_id` identifies the canonical logical attempt.
_Avoid_: group run ID, separate new native-attempt entity.
_Code_: src/milknado/domains/graph/_group_reservation.py:54.[^attempt-domain-source]

**Current owner** — The node ownership fence at the time of a graph operation.
Direct review redispatch can replace it without creating another logical task attempt.
_Avoid_: assuming the original run ID remains the current owner.
_Code_: `node.owner_run_id`, src/milknado/domains/graph/_transitions.py:255-300.[^attempt-domain-source]

**Worker invocation**, in this execution context, retains its existing actual-worker meaning.
Review redispatch creates a new invocation without creating another logical task attempt.
A replacement lifeline remains a monitor replacement, not a new invocation.[^attempt-domain-design]
The current invocation must match the active ledger identity and linked source run.
A successor requires durable predecessor cleanup and no other live worker for the node.[^attempt-domain-binding]
_Code_: src/milknado/domains/graph/_task_attempt_binding.py (`bind_worker_invocation`).

### Execution-start evidence and unknown-start hold

**Protected release** — Existing admission permits a gated worker to proceed; it does not prove command execution.
_Avoid_: execution confirmation.
A missing row for a specified graph run identifier denies release.[^attempt-domain-binding]
_Code_: src/milknado/loop/_process_startup.py (`_release_linked_worker`, `_activate_worker`).

**Execution-start evidence** — Invocation-bound trusted evidence of actual command execution, confirmed startup failure, or unknown start.
The graph-owned observation binds the canonical attempt, current owner, active invocation, and linked source run.
_Avoid_: gate acknowledgment, thread creation, or pane creation as execution proof.[^attempt-domain-f4]

**Unknown-start hold** — Above 1, unresolved start evidence retains ownership and permits no failed-count increment or automatic retry.
The hold ends only when evidence resolves or existing cancellation or abandonment rules apply.
_Avoid_: retryable failure, exhausted budget.
Default 1 remains unchanged.[^attempt-domain-f4]

[^attempt-domain-design]: Issue #495 user decisions F-1–F-4; src/milknado/domains/graph/_task_attempt_models.py; src/milknado/domains/graph/_task_attempt_facade.py; src/milknado/domains/graph/_task_attempts.py. These definitions describe implementation source, not publication status.
[^attempt-domain-f2]: User F-2 approves one failed task execution per attempt, durable linkage, and no historical backfill. Run rows remain evidence, not independent accounting entities.
[^attempt-domain-f4]: User F-4 approves an unknown-start recovery hold only above 1, with no failed count or automatic retry. Gate release is separate from execution evidence.
[^attempt-domain-source]: Historical identity references use source checkout `7ee3e189b246d489f694c8885aeeb7f8d481351e`. Current attempt linkage uses src/milknado/domains/graph/_run_models.py and _run_persistence.py; source-run fencing uses src/milknado/domains/graph/_task_attempt_binding.py (`bind_worker_invocation`).

[^attempt-domain-repair]: src/milknado/domains/dispatch/lifecycle.py:90-164; src/milknado/domains/dispatch/async_run.py:79-118; src/milknado/domains/dispatch/reconcile.py:48-71,196-231.
[^attempt-domain-binding]: src/milknado/domains/graph/_task_attempt_binding.py:7-53; src/milknado/domains/graph/_task_attempt_facade.py:40-71; src/milknado/loop/_process_startup.py:51-116.

_Source: Issue #495 approved accounting and recovery policies, approved Cure guards, and verified implementation source · Updated: 2026-10-10 · Supersedes: shifted protected-release and source-fence citations; canonical ownership terms remain unchanged._
