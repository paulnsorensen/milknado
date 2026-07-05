---
kind: goal
slug: tmux-run-primitive
roadmap: milestone-0-3-0
created: 2026-07-01
prereqs: []
---
# tmux as a first-class run primitive

## Intent

Detached ralph runs are currently opaque subprocesses
(`start_new_session=True`): the only windows into a running agent are log
tails and poll tools. Firstmate runs every crew agent in a named tmux window
(`fm-<id>`), which buys three things at once: live visibility (watch the agent
work), an attach point (inspect a struggling run without killing it), and pane
liveness as a supervision signal.

Milknado should offer tmux as a dispatch substrate for ralph runs: a run
started with tmux enabled gets a named window/session tied to its `run_id`,
and `milknado attach <run>` (CLI) drops the user into it. The existing
detached-subprocess path remains the default for headless/CI environments —
tmux is an opt-in primitive, not a new requirement.

This is the visibility half of the firstmate-usability adoption track;
[[interactive-run-steering]] builds the steering half on top of it.

Open questions to resolve during design (not silently):

- Session topology: one tmux session per project with a window per run
  (firstmate's shape), or one session per run?
- Naming contract: window names must survive run-id length limits and be
  reconcilable after a server restart (scan-and-reattach on startup?).
- Does pane liveness feed [[zero-token-run-supervision]] as an additional
  signal, and if so, what is the precedence vs the runs-table state?
- Behavior when tmux is absent or the server runs outside a terminal —
  fail-closed at dispatch, or silent fallback to the detached path?

## Acceptance

- A ralph run can be dispatched into a named tmux window keyed to its
  `run_id`, opt-in via config or dispatch parameter; the default detached
  path is unchanged.
- `milknado attach <run>` (or equivalent) attaches to a running run's window;
  attaching to a finished or non-tmux run fails with a clear message.
- Run teardown handles the window lifecycle: a completed run's window is
  cleaned up (or preserved on failure for inspection — behavior documented
  and tested either way).
- Server restart reconciles: existing run windows are rediscovered, not
  orphaned or double-created.
- Works for both dispatch families that produce long-lived processes; the
  tmux path is covered by tests that do not require an interactive terminal
  in CI (tmux in control mode or a fake).
- `architecture/execution.md` documents the tmux substrate, naming contract,
  and fallback behavior.
