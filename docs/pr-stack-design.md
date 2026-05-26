# Per-Node PR Stacking Design

## Problem

`RunLoop` today has one completion mode: squash-rebase each completed node onto a flat
`feature_branch`. This collapses parallel work into one branch and blocks the
`/pr-stack-by-overlap` workflow — workers can't open PRs (they run as agents with no GitHub
credentials), so the orchestrator must do it post-run, but it currently has no per-node branches
to work with.

## Goal

Add opt-in `pr_stack=True` mode that:
1. On node completion: squash-commits the node's worktree branch and pushes it to remote, instead
   of rebasing onto `feature_branch`.
2. After the run loop ends: groups completed branches by file-overlap, and opens stacked PRs via
   `gh pr create`.

## Key Invariants

- **Default unchanged**: `pr_stack=False` uses the existing rebase-and-merge path.
- **Workers untouched**: no agent-side changes — only the orchestrator gains new behaviour.
- **Single new module**: `pr_stack.py` owns all grouping and PR creation.
- **Line budgets**: all files ≤ 300 lines, all functions ≤ 40 lines.

## Data Flow

```
Node completes (success)
  │
  ├─ pr_stack=False  →  executor.complete(node_id, feature_branch)      [existing]
  │                     squash + rebase + merge + remove worktree
  │
  └─ pr_stack=True   →  executor.stage_for_pr(node_id, feature_branch)  [new]
                        squash_and_commit (using feature_branch as merge-base)
                        git push origin <node-branch>
                        remove worktree
                        mark_done
                        return branch_name
                        → accumulated in loop._completed_branches: dict[int, str]

RunLoop.run() ends
  │
  ├─ pr_stack=False  →  nothing extra
  │
  └─ pr_stack=True   →  open_stacked_prs(completed_branches, graph, base_branch, project_root)
                        → group by file overlap (union-find on completed node IDs)
                        → topological order within each group (Mikado graph edges)
                        → gh pr create per node
                        → return list[StackedPr]
                        → stored in RunLoopResult.stacked_prs
```

## Stacking Algorithm

```
given: completed_node_ids, file_ownership[node_id] = {file1, file2, ...}

1. Build undirected overlap graph:
   for (i, j) in combinations(completed_node_ids, 2):
     if file_ownership[i] & file_ownership[j]:
       add_edge(i, j)

2. Find connected components via Union-Find → groups

3. For each group:
   a. Topological sort by Mikado graph edges (prerequisites first; tie-break by node_id)
   b. Stack: group[0] targets base_branch; group[k] targets group[k-1].branch

4. For each node in order:
   gh pr create --base <base> --head <branch> --title <desc> --body <desc>
   → record StackedPr
```

Nodes with no file overlap with any other completed node become singleton groups, each targeting
`base_branch` directly.

## File Changes

| File | Change |
|------|--------|
| `src/milknado/adapters/git.py` | Add `push_branch(branch, remote)` |
| `src/milknado/domains/common/protocols.py` | Add `push_branch` to `GitPort` |
| `src/milknado/domains/execution/pr_stack.py` | **NEW** — `StackedPr`, `build_overlap_groups`, `open_stacked_prs` |
| `src/milknado/domains/execution/executor.py` | Add `Executor.stage_for_pr` |
| `src/milknado/domains/execution/run_loop/_completion.py` | PR-stack branch in `handle_completion` |
| `src/milknado/domains/execution/run_loop/__init__.py` | `_pr_stack`, `_completed_branches`, `pr_stack=` param |
| `src/milknado/domains/execution/run_loop/_result.py` | Add `stacked_prs` field |
| `src/milknado/domains/common/config.py` | Add `pr_stack: bool = False` |
| `src/milknado/cli_run.py` | Thread `pr_stack` into `loop.run()` |

## What's Not In Scope

- Conflict resolution between stacked branches (human merge task after opening PRs)
- Updating PR bases if a lower PR is merged (GitHub's "Update branch" handles it)
- Non-GitHub remotes (`gh` CLI is GitHub-specific)
- Automatic PR merge
