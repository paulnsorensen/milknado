# run_inline isolates per dispatch — ISOLATE by default, merge-back on success

`milknado_run_inline` / `milknado_run_inline_start` take a `worktree` argument
(a `WorktreeMode` enum) that the caller sets per dispatch. It is **safe by
default**:

- **`ISOLATE` (default)** — the dispatch creates a git worktree + branch
  (`_create_node_worktree`, `isolate.py`, honoring `worktree_pattern`), runs
  the worker with `cwd = worktree_path`, and records `worktree_path` on the node.
  The diff lands on the isolated branch, not the caller's checkout.
- **`THIS_BRANCH`** — the worker runs in the shared checkout (`cwd =
  project_root`), with `worktree_path` left unset (default) — `set_worktree` is
  ISOLATE-only. The explicit opt-in for "the
  diff lands in my current working tree" — the old, pre-#171 behavior.

## merge_back (ISOLATE only)

A second, orthogonal argument `merge_back: bool = True` decides the ISOLATE
branch's fate on a clean (exit 0) run:

- **`True` (default)** — rebase-merge the isolated branch back into the caller's
  dispatch branch, then tear the worktree down. This REUSES the executor's
  `WorktreeManager.rebase_and_merge` (`execution/executor.py:288`) — the same
  squash → rebase → fast-forward → fail-closed-teardown machinery the ralph loop
  uses — rather than reimplementing it. A teardown that refuses (dirty/unlanded
  work) preserves the worktree on disk and surfaces its path as
  `worktree_preserved`. Net: "safely isolated during the run, result merged back
  to my branch on success" — the safe successor to old `run_once`'s
  diff-lands-here.
- **`False`** — leave the isolated branch/worktree for the caller to review or PR.
  Teardown while the run is active happens via `milknado_run_cancel`; once the run
  is terminal, `milknado_run_cancel` no-ops, so a completed `merge_back=False`
  worktree must be removed manually (`git worktree remove`).

`THIS_BRANCH` ignores `merge_back` (there is nothing to merge). The dispatch
branch a merge-back lands on is the caller's current branch
(`GitAdapter.current_branch()`), resolved at dispatch time.

The sync path (`dispatch_node_sync`) merges back inline after the worker returns;
the async path (`_async_worker`) defers the merge-back to the worker thread, after
the worker process exits, since the start tool returns immediately. On exit 0 the
node is marked terminal DONE and a second dispatch is refused (reset the node's
status to retry).

## Why ISOLATE is the default (2026-07-04 incident)

Verified empirically 2026-07-04: three concurrent one-shot workers (tilth n14,
easy-cheese n10, hallouminate n6) all ran in their repos' main checkouts under the
old shared-checkout behavior. Consequences observed the same night:

- **Cross-contamination** — the tilth worker collided with an unrelated concurrent
  cherry-pick in the same tree; a blind `git stash` inside the worker tangled with
  the other actor's index.
- **Work displaced by other actors** — the easy-cheese worker's uncommitted diff
  was `git stash`-ed away by a concurrent session clearing the tree for its own
  commit; the diff looked destroyed until forensics found it in `stash@{0}`.

Those incidents are why the default flipped to ISOLATE: concurrent or unattended
dispatch is safe out of the box, and `THIS_BRANCH` is the deliberate opt-in for
"I want the diff in this tree." The teardown side of the same safety story is
governed by [[fail-closed-worktree-teardown]].
