---
kind: goal
slug: interactive-run-steering
roadmap: milestone-0-3-0
created: 2026-07-01
prereqs: [tmux-run-primitive]
---
# Non-headless agents: steer a running run mid-flight

## Intent

Every milknado worker today is headless: once dispatched, the only human
levers are cancel or wait. Firstmate's most-loved UX affordance is the
opposite — every crew agent runs interactively in its tmux window, and the
captain can type into it mid-run to redirect, answer a question, or unblock
it, without restarting the task.

With [[tmux-run-primitive]] providing the window, this goal makes the agent
inside it steerable: a run dispatched in interactive mode runs the harness in
its normal (non-headless) mode, so a human attaching to the window can
converse with the worker directly. The ralph loop contract still holds — the
run ends when quality gates pass — but a human can participate in getting it
there.

This is the second half of the firstmate-usability adoption track. Together
with ralph trees and wiki roadmaps, it is what lets milknado be the golden
path over firstmate: the same casual working-with-sub-agents feel, backed by
a real dependency graph, deterministic batching, and hard verification gates
instead of prose rules.

Open questions to resolve during design (not silently):

- Harness support matrix: which of the allowlisted harnesses (claude, codex,
  cursor-agent, gemini) can run interactively under tmux with the worker
  hook still firing? Interactive mode may be per-harness opt-in.
- Loop semantics: in interactive mode, does the ralph iteration boundary
  survive (harness exits, loop re-invokes), or does the loop become a single
  long-lived interactive session with gate checks on idle?
- How does human input mid-run interact with the run's brief/RALPH.md
  contract — is a human redirection recorded (e.g. as a `run_messages` row)
  so the audit trail stays truthful?
- Supervision interplay: does typing into a window suppress
  [[zero-token-run-supervision]] stall detection for that run?

## Acceptance

- A run can be dispatched in interactive mode (opt-in, per-run), launching
  the harness non-headless inside the run's tmux window; headless remains
  the default.
- A human attaching to the window can send input to the running agent and
  the agent acts on it, demonstrated by an e2e or scripted-tmux test.
- Quality-gate verification is unchanged: an interactive run still cannot
  reach DONE without a passing `milknado_node_verify` — human participation
  does not weaken the gate.
- Human mid-run input leaves a durable trace (documented mechanism, e.g. a
  `run_messages` row or status event), so a harvested run shows whether it
  was steered.
- Unsupported-harness dispatch in interactive mode fails fast with a clear
  message listing supported harnesses.
- `architecture/execution.md` documents interactive mode, its harness
  matrix, and its interaction with supervision and verification.
