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

## Review checks retained from the historical campaign

Reject broad type-check suppressions that hide unrelated errors.
Check that tests and assertions remain in place.
Check that concurrency tests retain their barriers.
Verify claimed fixes in the diff and executable tests.
Do not accept an agent's summary as the only evidence.
