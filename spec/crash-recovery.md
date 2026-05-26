# Crash Recovery Design

## Problem

A `milknado run` interrupted by Ctrl-C (or a harder crash) leaves three failure modes:

1. **Orphaned workers** — Ralph subprocesses keep merging in their worktrees for minutes
   after the parent process exits.
2. **Stuck-RUNNING nodes** — DB says `RUNNING` but no process is driving them to
   completion. On the next `milknado run`, `get_ready_nodes` skips all nodes whose
   children are still `RUNNING`, so the graph is effectively locked.
3. **Split git state** — A node's work may be partially committed or rebased but the
   DB never reached `DONE`.

## Failure Scenarios

| Scenario | DB state | Worktree | Branch |
|---|---|---|---|
| Ctrl-C during poll loop, worker still running | RUNNING | exists | exists (no squash yet) |
| Ctrl-C during `complete()` before `mark_done` | RUNNING | removed | exists (squash done, rebased) |
| Hard crash (OOM/kill) mid-dispatch | RUNNING | may exist | may exist |

## Design

### (a) Graceful SIGINT Shutdown

**Where**: `RunLoop._execute_run`, existing `except KeyboardInterrupt` block.

**What**: Before re-raising, for each active run:
1. Call `ralph.stop_run(run_id)` — stops the worker subprocess (best-effort).
2. Call `executor.fail(node_id)` — cleans the worktree, marks `FAILED`.
3. Call `graph.mark_pending(node_id)` — resets to `PENDING` so the next plain
   `milknado run` picks them up without `--resume`.

Failure in any step is logged and skipped; the loop continues to the next node.
`self._active` is cleared at the end so the caller's telemetry sees 0 active runs.

### (b) Reconcile: `milknado run --resume`

**New flag**: `milknado run --resume` runs reconciliation before the normal dispatch loop.

**New module**: `src/milknado/domains/execution/reconcile.py`

Algorithm (`reconcile_stale_nodes(graph, git, ralph, feature_branch)`):

For each node with `status == RUNNING`:

1. **Stop Ralph run** (best-effort via `ralph.stop_run`).
2. **Check if work was already merged** (`_appears_merged`):
   - Returns `False` if `node.branch_name` is absent.
   - Returns `False` if `node.worktree_path` still exists on disk (worker is still
     mid-run or merge never happened).
   - Returns `True` if `git.branch_exists(branch)` and
     `git.is_ancestor(branch, feature_branch)` — the squash commit is reachable from
     the feature branch, meaning `rebase_and_merge` completed but `mark_done` did not.
3. **If merged** → `graph.mark_done(node_id)`.
4. **Otherwise** → clean worktree (best-effort), `graph.mark_pending(node_id)`.

**Return**: `ReconcileResult(resolved_done: list[int], reset_pending: list[int])`.

### (c) Idempotent Re-dispatch

`GitAdapter.create_worktree` already uses `git worktree add -B` (create-or-reset),
so re-dispatching a previously-dispatched node never fatally collides on an existing
branch. No additional changes required.

For the "work already done but node reset to PENDING" case: when re-dispatched, the
Ralph agent runs again in a fresh worktree. `squash_and_commit` finds an empty staged
diff (work is already on the feature branch) and skips the commit. The node reaches
`DONE` via the normal completion path.

## New Git Operations

Two new methods on `GitAdapter` (and `GitPort` protocol):

```python
def branch_exists(self, branch: str) -> bool:
    # git rev-parse --verify refs/heads/<branch>
    ...

def is_ancestor(self, ref: str, of_ref: str) -> bool:
    # git merge-base --is-ancestor <ref> <of_ref>
    ...
```

`is_ancestor(ref, of_ref)` returns `True` when `ref` is reachable from `of_ref` (i.e.
`ref` was merged into `of_ref`).

## Files Changed

| File | Change |
|---|---|
| `src/milknado/adapters/git.py` | Add `branch_exists`, `is_ancestor` |
| `src/milknado/domains/common/protocols.py` | Add to `GitPort` protocol |
| `src/milknado/domains/execution/reconcile.py` | New — reconcile logic |
| `src/milknado/domains/execution/__init__.py` | Export `ReconcileResult`, `reconcile_stale_nodes` |
| `src/milknado/domains/execution/run_loop/__init__.py` | Graceful SIGINT shutdown |
| `src/milknado/cli_run.py` | Add `--resume` flag |
| `tests/test_adapters_git.py` | Tests for new git methods |
| `tests/test_execution.py` | Tests for SIGINT shutdown |
| `tests/test_reconcile.py` | New — tests for reconcile |

## Non-Goals

- Replaying partially-written worktree changes (re-dispatch is safer and already idempotent).
- Detecting split-commit state within a worktree (test committed, impl not).
- Auto-merging unfinished node branches into the feature branch.
