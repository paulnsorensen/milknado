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

import fcntl
from contextlib import contextmanager
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


def _create_node_worktree(  # noqa: ANN001, ANN202
    git, root, node_id: int, description: str, worktree_pattern: str
):
    """Create the node's durable worktree under project_root, mirroring the executor.

    Formats ``worktree_pattern`` ({node_id}, {slug}) for the directory name and
    uses the executor's branch convention (milknado/<id>-<slug>), reusing
    GitAdapter.create_worktree with the same path-traversal guard. Slug derivation
    reuses the executor's ``_slugify`` (lazy-imported to keep the dispatch slice
    free of the execution slice's import weight).
    """
    from milknado.domains.execution.executor import _slugify

    slug = _slugify(description)
    wt_path = root / worktree_pattern.format(node_id=node_id, slug=slug)
    resolved_root = root.resolve()
    if not wt_path.resolve().is_relative_to(resolved_root):
        raise ValueError(f"worktree path resolves outside project_root: {wt_path!r}")
    branch = f"milknado/{node_id}-{slug}"
    git.create_worktree(wt_path, branch)
    return wt_path, branch


def create_isolated_worktree(root: Path, node_id: int, description: str, worktree_pattern: str):  # noqa: ANN201
    """Create the node's isolated worktree + branch; return ``(wt_path, branch)``.

    Reuses ``_create_node_worktree`` (the same helper the native claim path uses)
    so the path/branch convention and the path-traversal guard live in one place.
    """
    from milknado.adapters import GitAdapter

    return _create_node_worktree(GitAdapter(root), root, node_id, description, worktree_pattern)


@contextmanager
def _merge_back_lock(root: Path):  # noqa: ANN202
    """Serialize merge-backs across processes on a repo-scoped ``flock``.

    Both the sync (``_maybe_merge_back``) and async (``_async_merge_back``) paths
    funnel through ``merge_back_isolated``, so a single exclusive lock here is the
    one choke point that stops two concurrent dispatches from rebase-merging onto
    the same dispatch branch at once. Blocking acquire; the lock releases when the
    file handle closes.
    """
    lock_dir = root / ".milknado"
    lock_dir.mkdir(parents=True, exist_ok=True)
    with (lock_dir / "merge-back.lock").open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


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
    # Re-resolve the dispatch branch at merge time rather than trusting the branch
    # captured when the context was built: fast_forward targets root's live current
    # branch, so the rebase target must agree with it, not a stale snapshot.
    feature_branch = resolve_feature_branch(root)
    with _merge_back_lock(root):
        try:
            result = manager.rebase_and_merge(
                ctx.worktree_path,
                feature_branch,
                ctx.node_id,
                ctx.description,
                worker_branch=ctx.worker_branch,
            )
        except UnlandedWorkError:
            # The branch never landed on the dispatch branch, so this is not a
            # successful rebase: report rebased=False and preserve the worktree.
            return MergeBackResult(rebased=False, worktree_preserved=str(ctx.worktree_path))
    preserved = None if result.success else str(ctx.worktree_path)
    return MergeBackResult(rebased=result.success, worktree_preserved=preserved)


def resolve_feature_branch(root: Path) -> str:
    """The dispatch branch an ISOLATE merge-back lands on: the caller's current branch."""
    from milknado.adapters import GitAdapter

    return GitAdapter(root).current_branch()
