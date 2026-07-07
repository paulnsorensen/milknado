---
kind: goal
slug: worker-coordinator-messaging
roadmap: milestone-0-3-0
created: 2026-06-30
prereqs: [coordinator-run-entity]
---
# Build worker↔coordinator messaging on run_messages

## Intent

The deposit channel (`architecture/execution.md` §Deposit channel) added
`run_messages` — an append-only per-run message table with atomic seq assignment
— but shipped only the `role='result'` deposit path. The page records that
**progress messaging and coordinator→worker inbound are deliberately NOT built;
the `run_messages` shape permits them later (YAGNI — spec non-goal)**. With the
native backend hardened and a real coordinator identity in place, the inbound
direction now has a concrete addressee, so the channel is worth completing.

Done looks like: a worker can post incremental **progress** messages mid-run
(surfaced by the poll tools alongside the existing `result` and log-tail
`summary`), and a coordinator can send **inbound** messages a worker can read —
the bidirectional channel the table was shaped for. Depends on
[[coordinator-run-entity]] because inbound messaging addresses a coordinator
identity; without a real one, "coordinator→worker" has no sender to attribute.

## Acceptance

- A worker can append `role='progress'` (or equivalent) messages during a run via
  an MCP tool, using the same atomic single-statement seq assignment the result
  path uses (concurrent depositors cannot collide).
- `milknado_run_inline_poll` / `milknado_run_loop_poll` surface progress messages
  distinctly from the final `result` and the log-tail `summary`.
- A coordinator→worker inbound path exists: the coordinator writes an inbound
  message and the worker can read it; the worker tool allowlist
  (`WORKER_ALLOWED_TOOLS`) is extended only as far as the new direction requires,
  with the Family-4 coordinator-only prohibition preserved.
- The `MILKNADO_RUN_ID`-gated soft-no-op behavior is preserved: a worker with no
  running run row (e.g. a DONE-node re-run) must not raise "run not found" when it
  posts.
- New paths are tested per branch (poll-surfacing, concurrent append, inbound
  read, no-run-row no-op) to satisfy `codecov/patch`.
- `architecture/execution.md` §Deposit channel is updated to mark progress +
  inbound as built and document the new tools.
