---
kind: goal
slug: coordinator-run-entity
roadmap: milestone-0-3-0
created: 2026-06-30
prereqs: [broaden-ultracode-e2e]
---
# Introduce a real coordinator-run identity

## Intent

Goal-claim fencing (`architecture/execution.md` §Goal-claim fencing, §Run
identity) currently anchors the same-coordinator exemption on
`claim["pid"] == os.getpid()`, because under the one-process-per-worktree model
the server pid *is* the coordinator. The wiki names this a deliberate proxy with
known limits: it dies with the process, breaks under pid reuse (fails safe, but
spuriously), and cannot represent a coordinator that respawns or a multi-run
server. The architecture already carries the seams for the real fix —
`ancestor_goal_claimed_by_other`'s `caller_run_id` parameter and the
`release_goal` method have **zero production callers** and are explicitly
preserved as "hooks for a future coordinator-run entity" that must not be deleted
as dead code.

This goal makes that entity real: a concrete coordinator-run identity that spans
the sibling dispatches a coordinator makes while driving a goal, so claim
ownership moves off the pid proxy onto a threaded identity. Done looks like:
goal claims survive a coordinator respawn, a multi-run server is representable,
and the previously-unwired seams have production callers. Per the wiki's standing
rule, the design must **name which concrete identity** owns a claim — this goal
creates the third option (the coordinator-run entity) the rule anticipates.

Depends on [[broaden-ultracode-e2e]]: changing the claim-ownership model touches
both the subprocess and native dispatch paths, so a validated e2e baseline must
exist first to catch regressions in fencing and same-coordinator exemption.

## Acceptance

- A coordinator-run identity exists that spans sibling task dispatches under one
  goal; the same-coordinator goal-claim exemption is keyed on it rather than on
  `os.getpid()`.
- `caller_run_id` (in `ancestor_goal_claimed_by_other`) and `release_goal` gain
  real production callers — the seams are wired, not removed.
- A coordinator respawn (new process, same logical coordinator run) no longer
  fences itself out of its own goal's sibling dispatches.
- The atomic-claim mutex invariant is preserved: pid/identity rides in the
  `INSERT OR IGNORE` statement (no two-step insert-then-set window), and pid
  reuse / stale claims still fail *safe* (spurious refusal, never false-allow).
- The pre-existing fencing tests still pass, and new tests cover the
  respawn-survives-claim and multi-run-representable cases (per-branch coverage
  for `codecov/patch`).
- `architecture/execution.md` is updated: the "intentionally unwired seams" note
  and the run-identity §Rule are revised to describe the now-real entity.
