# ADR — Insert runs row at ralph dispatch in Executor (run-lifecycle-robustness-001)

Date: 2026-07-23 · Status: accepted · Spec: run-lifecycle-robustness

## Decision

`Executor._create_ralph_run` calls `graph.start_run(node_id, run_id, pid, log_path, command)` after `RalphLauncher.start_run` succeeds, mirroring the inline-path insert shape (`dispatch/lifecycle.py:121`). On cancel, the graph row is marked cancelled when present.

## Rationale

- The ralph dispatch path (`Executor` → `LoopAdapter` → in-memory `RunManager`) never inserted a `runs` row, so review-verdict `run_messages` inserts FK-failed and verdicts were lost (#296); `run_list`/`run_loop_poll` were blind to ralph runs.
- Executor already owns dispatch orchestration and holds the graph handle — the insert belongs with the other callers (`app/node.py:109`, `app/ralph.py:220`, `dispatch/async_run.py:351`).

## Alternatives

- Upsert a placeholder row inside `deposit_run_message` — smallest diff, but rows carry no node/pid/log linkage truth and `run_list` stays blind.
- Push a graph handle into `LoopAdapter`/`RunManager` so ralph runs persist themselves — gives the adapter layer a DB dependency it deliberately does not have.
- Do Nothing — accept verdict loss and error spam; contradicts the filed bug.

## Consequences

Ralph runs become listable, pollable, and message-depositable like inline runs. Executor gains one more graph call at dispatch; cancel path must tolerate a missing row for pre-fix runs.
