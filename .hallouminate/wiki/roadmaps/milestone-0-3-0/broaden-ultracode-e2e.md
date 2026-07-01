---
kind: goal
slug: broaden-ultracode-e2e
roadmap: milestone-0-3-0
created: 2026-06-30
prereqs: [ultracode-install-ergonomics]
---
# Broaden ultracode end-to-end coverage

## Intent

The native backend's live verification (`history/workflow-executor-decision.md`
§Verification) proved the *concept* end-to-end — but only on a single scratch
node, in `single` mode, on the happy path: `claim` → worker `deposit_result` →
`node_verify ok=true` → gated `set_status(done)`. The riskier shapes are
unproven live: **mode B (cold re-dispatch / `redispatch`) loop**, **multi-node
parallel fan-out** across a real batch, and **gate-failure paths** where
`node_verify` returns `ok=false` and the orchestrator-side verify→re-dispatch
loop must drive the node to completion or terminal failure.

Done looks like: a repeatable e2e harness that exercises the native backend
across these shapes and asserts the right terminal states, so a regression in the
claim/verify/gate machinery is caught rather than discovered in production. This
depends on [[ultracode-install-ergonomics]] — the harness should drive the same
installable entry point a user would.

## Acceptance

- An e2e test (or documented reproducible run) covers, against real nodes, at
  minimum: (a) `redispatch` mode driving a node through more than one iteration;
  (b) a multi-node batch fanned out in parallel via the workflow; (c) a
  gate-failure path where `node_verify` returns `ok=false` and the node ends in
  the correct terminal state (re-dispatched to success, or `failed`).
- Each scenario asserts the *terminal* graph state and the gated-completion
  invariant (a native-claimed node reaches `done` only after a passing verify) —
  not just that the run started.
- The server-side completion gate's fail-closed behavior is exercised: a
  native-claimed node cannot be marked `done` without a passing `node_verify`.
- Results are recorded back into `workflow-executor-decision.md` §Verification so
  the "only one scratch node, happy path only" caveat is retired with evidence.
