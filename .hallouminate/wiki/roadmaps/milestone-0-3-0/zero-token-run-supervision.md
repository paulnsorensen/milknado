---
kind: goal
slug: zero-token-run-supervision
roadmap: milestone-0-3-0
created: 2026-07-01
prereqs: [worker-coordinator-messaging]
---
# Zero-token event-driven supervision of ralph runs

## Intent

Today a coordinator supervising detached ralph runs burns model turns polling
`milknado_run_loop_poll` / `milknado_run_list` — every check spends tokens even
when nothing changed. Firstmate demonstrates the alternative: a bash watcher
(`fm-watch.sh`) sleeps on the fleet, classifies wake events in bash at zero LLM
cost, absorbs benign wakes (heartbeats, no-change progress) silently, and wakes
the orchestrating agent only on actionable transitions.

Milknado should supervise the same way. A watcher process (bash or Python, no
LLM) observes run state — the `runs` table, `run_messages` (including the
`role='progress'` messages [[worker-coordinator-messaging]] adds, which is why
that goal is a prerequisite), and worktree/process liveness — classifies each
transition as benign or actionable, and delivers actionable events to the
coordinator through a durable queue that survives a missed wake (firstmate's
`state/.wake-queue` pattern). Benign transitions never reach a model.

Done looks like: a coordinator driving N parallel ralph runs makes zero
polling tool calls while runs are healthy, and is woken promptly when a run
finishes, fails a quality gate, dies, or stalls.

Open questions to resolve during design (not silently):

- Delivery mechanism for the wake: how does a headless watcher "wake" a
  coordinator agent across the supported harnesses (hook? stdin injection?
  MCP notification? file the coordinator checks on its own turn-end)?
- Where does the actionable/benign classification table live — hardcoded,
  `milknado.toml`, or per-run config at dispatch time?
- Does the watcher belong to the MCP server process, or is it a detached
  sibling like the ralph runs themselves (surviving server restart)?
- Stall detection: what is the signal for "running but stuck" when there is
  no tmux pane to inspect (log mtime? progress-message age)?

## Acceptance

- A watcher process exists that observes run lifecycle state without any LLM
  involvement, and its wake classification (actionable vs benign) is
  deterministic and unit-tested per class of transition.
- Actionable events land in a durable queue that is drained on coordinator
  wake; an event raised while the coordinator is absent is not lost.
- A demo/e2e scenario shows a coordinator supervising ≥2 parallel ralph runs
  to completion with zero `run_loop_poll` calls during healthy execution.
- Dead-process and stalled-run detection each produce an actionable wake,
  covered by tests.
- The polling tools remain available and unchanged (the watcher is additive;
  polling stays the fallback for harnesses without a wake channel).
- `architecture/execution.md` documents the watcher, the classification
  table, and the wake-delivery mechanism per harness.
