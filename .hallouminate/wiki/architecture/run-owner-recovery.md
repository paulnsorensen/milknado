# Run owner recovery

Normal detached polling reads durable run state and logs; it must not query tmux. tmux remains a launch and explicit inspection/recovery mechanism.[^1]

A NULL `runs.pid` is not evidence that a run died: executor-dispatched work uses the fenced node PID as the coordinator liveness identity. Recovery finalizes a run only after a known owner PID is confirmed dead; absent optional identity metadata keeps the existing timeout path.[^2]

Cancellation first proves an owner can be stopped or is already terminal. A refusal leaves the run and cancellation marker untouched, preventing a delayed cooperative cancellation from mutating a still-running coordinator-owned run.[^3]

[^1]: src/milknado/mcp/run.py; src/milknado/mcp/ralph.py
[^2]: src/milknado/domains/dispatch/reconcile.py
[^3]: src/milknado/domains/dispatch/cancel.py