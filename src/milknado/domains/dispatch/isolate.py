"""ISOLATE-mode worktree creation + merge-back for one-shot run_inline dispatch.

run_inline's ISOLATE mode runs a worker in its own git worktree/branch and, when
``merge_back`` is set, rebase-merges the branch back into the caller's dispatch
branch on exit 0. The merge-back REUSES the executor's
``WorktreeManager.rebase_and_merge`` — the same squash → rebase → fast-forward →
fail-closed-teardown machinery the ralph loop uses — rather than reimplementing
it. WorktreeManager and the git adapter are imported lazily so importing the
dispatch slice does not drag in the execution slice's TUI dependencies (the same
lazy pattern ``cancel.py`` uses).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class IsolateContext:
    """The state a deferred (async) merge-back needs after the worker exits."""

    worktree_path: Path
    worker_branch: str
    feature_branch: str
    node_id: int
    description: str


@dataclass(frozen=True)
class MergeBackResult:
    rebased: bool
    worktree_preserved: str | None


def create_isolated_worktree(root: Path, node_id: int, description: str, worktree_pattern: str):  # noqa: ANN201
    """Create the node's isolated worktree + branch; return ``(wt_path, branch)``.

    Reuses ``_create_node_worktree`` (the same helper the native claim path uses)
    so the path/branch convention and the path-traversal guard live in one place.
    """
    from milknado.adapters import GitAdapter
    from milknado.mcp_node import _create_node_worktree

    return _create_node_worktree(GitAdapter(root), root, node_id, description, worktree_pattern)


def setup_isolated_worktree(graph, root: Path, node, run_id: str, worktree_pattern: str):  # noqa: ANN001, ANN201
    """Create the node's ISOLATE worktree + branch and record it on the run_id-fenced node.

    The setup half shared by the sync (``dispatch_node_sync``) and async
    (``run_inline_start``) dispatch paths; the async side additionally layers an
    ``IsolateContext`` on the returned ``(wt_path, worker_branch)`` for the
    deferred merge-back.
    """
    wt_path, worker_branch = create_isolated_worktree(
        root, node.id, node.description, worktree_pattern
    )
    graph.set_worktree(node.id, run_id, str(wt_path), worker_branch)
    return wt_path, worker_branch


def merge_back_isolated(root: Path, ctx: IsolateContext) -> MergeBackResult:
    """Rebase-merge an ISOLATE branch back into its dispatch branch, then tear the
    worktree down fail-closed. Delegates to ``WorktreeManager.rebase_and_merge``.

    A teardown that refuses (dirty/unlanded work, ``UnlandedWorkError``) preserves
    the worktree on disk and returns its path as ``worktree_preserved``; a rebase
    that did not land leaves the worktree in place for inspection likewise.
    """
    from milknado.adapters import GitAdapter
    from milknado.domains.common.errors import UnlandedWorkError
    from milknado.domains.execution import WorktreeManager

    manager = WorktreeManager(GitAdapter(root))
    try:
        result = manager.rebase_and_merge(
            ctx.worktree_path,
            ctx.feature_branch,
            ctx.node_id,
            ctx.description,
            worker_branch=ctx.worker_branch,
        )
    except UnlandedWorkError:
        return MergeBackResult(rebased=True, worktree_preserved=str(ctx.worktree_path))
    preserved = None if result.success else str(ctx.worktree_path)
    return MergeBackResult(rebased=result.success, worktree_preserved=preserved)


def resolve_feature_branch(root: Path) -> str:
    """The dispatch branch an ISOLATE merge-back lands on: the caller's current branch."""
    from milknado.adapters import GitAdapter

    return GitAdapter(root).current_branch()
