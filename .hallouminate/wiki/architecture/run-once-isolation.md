# run_once dispatches in the shared checkout — no worktree

`milknado_run_once` / `milknado_run_once_start` never create a git worktree.
The dispatch chain (`mcp_run.py` → `start_headless_async` → `_spawn_worker`,
`runner.py:157`) spawns the worker with `cwd=str(project_root)` — the shared
main checkout — unconditionally. No config key changes this; the
`worktree_pattern` setting is consumed only by the `milknado_todo_claim` /
run-loop path, which does isolate (`mcp_node.py:136`, `_create_node_worktree`).
Nodes dispatched via run_once leave `worktree_path` as `NULL` in the db — the
run_once path claims via `claim_node`, which never writes the column (the schema
default is NULL, `worktree_path TEXT` with no DEFAULT).

Verified empirically 2026-07-04: three concurrent `run_once_start` workers
(tilth n14, easy-cheese n10, hallouminate n6) all ran in their repos' main
checkouts. Consequences observed the same night:

- **Cross-contamination** — the tilth worker collided with an unrelated
  concurrent cherry-pick in the same tree (its own log flagged the contention,
  and a blind `git stash` inside the worker tangled with the other actor's
  index).
- **Work displaced by other actors** — the easy-cheese worker's uncommitted
  diff was `git stash`-ed away by a concurrent session clearing the tree for
  its own commit; the diff looked destroyed until forensics found it in
  `stash@{0}`.

Rule of thumb until this changes: **run_once is only safe when nothing else
touches that checkout for the run's duration.** For concurrent or unattended
dispatch, use the claim/run-loop path (isolated worktrees) or instruct workers
to commit onto a branch immediately. Whether run_once's non-isolation is
intentional (cheap one-off dispatch) or a gap to close is an open design
question — adjacent to [[fail-closed-worktree-teardown]], which governs the
teardown side of the same safety story.
