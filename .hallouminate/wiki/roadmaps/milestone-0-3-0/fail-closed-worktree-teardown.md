---
kind: goal
slug: fail-closed-worktree-teardown
roadmap: milestone-0-3-0
created: 2026-07-01
prereqs: []
---
# Fail-closed worktree teardown

## Intent

Every worktree removal today goes through `GitAdapter.remove_worktree`, which
runs `git worktree remove --force` unconditionally (`adapters/git.py:57-58`).
All four call sites — the orphan prune in `milknado_run_loop_start`,
`WorktreeManager.ensure_clean`, `.remove`, and `.discard` — inherit that
behavior: uncommitted or unlanded work in a failed, cancelled, or reclaimed
run's worktree is destroyed silently. Firstmate's `fm-teardown.sh` sets the
bar this goal adopts: refuse to reap a worktree that is dirty or whose
commits have not landed in the target branch, with an explicit containment
check (handling squash-merge, where the branch tip is not an ancestor of the
target).

Done looks like: teardown is fail-closed by default — a dirty or unlanded
worktree survives, with a clear diagnostic naming what is at risk — and
destruction is always an explicit, separately-named act (a `discard`-style
call or flag), never the default path. This matters more once
[[interactive-run-steering]] exists: a human's mid-run work in a steered
worktree becomes destroyable by the same `--force`.

Open questions to resolve during design (not silently):

- Landed-work check: ancestor containment is wrong under rebase-merge and
  squash-merge — patch-id comparison, `git cherry`, or PR-head containment
  (firstmate's approach)? Pick one and document its false-refusal modes.
- The orphan-prune path in `milknado_run_loop_start` exists so
  `git worktree add` can reuse the path — if a dirty orphan now refuses
  removal, what unblocks re-dispatch (relocate the orphan? new worktree
  path per run)?
- Does refusal fail the calling operation or degrade it (dispatch proceeds
  in a fresh path, refusing worktree logged and kept)?

## Acceptance

- Default teardown refuses to remove a worktree with uncommitted changes or
  commits not landed in the target branch; the refusal names the worktree
  and what is unlanded/dirty.
- Squash- and rebase-merged branches are recognized as landed (no false
  refusal for the normal ralph rebase-merge flow), covered by tests for
  each merge shape.
- Forced destruction remains available only through an explicitly-named
  discard path; no caller reaches `--force` implicitly.
- The orphan-reclaim dispatch path still works when the orphan is clean, and
  has a documented, tested behavior when the orphan is dirty.
- All four existing call sites are audited and re-pointed at the fail-closed
  path or the explicit discard path, each with a stated rationale.
- `architecture/execution.md` documents the teardown contract and the
  landed-work check chosen.
