"""ISOLATE-mode worktree creation + merge-back for one-shot run_inline dispatch.

ISOLATE mode runs a worker in its own git worktree/branch. When ``merge_back``
is enabled, this module performs the direct GitPort merge-back on exit 0:
squash, rebase, compare-and-swap the captured target ref, and remove the
worktree only after the landing succeeds. Failed or mismatched operations
preserve the worktree for inspection.
"""

from __future__ import annotations

import fcntl
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from milknado.domains.common import GitPort, UnlandedWorkError, slugify
from milknado.domains.common.errors import GitOperationError


@dataclass(frozen=True)
class IsolateContext:
    """The state a deferred (async) merge-back needs after the worker exits."""

    worktree_path: Path
    worker_branch: str
    target_branch: str
    base_oid: str
    node_id: int
    description: str


@dataclass(frozen=True)
class MergeBackResult:
    rebased: bool
    worktree_preserved: str | None


def _create_node_worktree(
    git: GitPort,
    root: Path,
    node_id: int,
    description: str,
    worktree_pattern: str,
) -> IsolateContext:
    slug = slugify(description) or "node"
    wt_path = root / worktree_pattern.format(node_id=node_id, slug=slug)
    if not wt_path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"worktree path resolves outside project_root: {wt_path!r}")
    target_branch = git.current_branch()
    target_ref = f"refs/heads/{target_branch}"
    base_oid = git.resolve_ref(target_ref)
    worker_branch = f"milknado/{node_id}-{slug}"
    git.create_worktree(wt_path, worker_branch)
    if git.resolve_ref(worker_branch) != base_oid:
        git.force_remove_worktree(wt_path)
        raise GitOperationError("checkout changed while isolated worktree was being created")
    return IsolateContext(
        worktree_path=wt_path,
        worker_branch=worker_branch,
        target_branch=target_branch,
        base_oid=base_oid,
        node_id=node_id,
        description=description,
    )


def create_isolated_worktree(
    git: GitPort,
    root: Path,
    node_id: int,
    description: str,
    worktree_pattern: str,
) -> IsolateContext:
    return _create_node_worktree(git, root, node_id, description, worktree_pattern)


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


def setup_isolated_worktree(
    graph,  # noqa: ANN001
    git: GitPort,
    root: Path,
    node,  # noqa: ANN001
    run_id: str,
    worktree_pattern: str,
) -> IsolateContext:
    context = create_isolated_worktree(git, root, node.id, node.description, worktree_pattern)
    graph.set_worktree(node.id, run_id, str(context.worktree_path), context.worker_branch)
    return context


def merge_back_isolated(git: GitPort, root: Path, ctx: IsolateContext) -> MergeBackResult:
    target_ref = f"refs/heads/{ctx.target_branch}"
    with _merge_back_lock(root):
        try:
            current_target = git.current_branch()
            if current_target != ctx.target_branch:
                raise GitOperationError(
                    "merge-back target",
                    f"caller requested {ctx.target_branch!r}, checkout is {current_target!r}",
                )
            git.squash_and_commit(
                ctx.worktree_path,
                ctx.base_oid,
                f"feat: complete node {ctx.node_id} — {ctx.description}",
            )
            rebased = git.rebase(ctx.worktree_path, ctx.base_oid)
            if not rebased.success:
                return MergeBackResult(rebased=False, worktree_preserved=str(ctx.worktree_path))
            worker_oid = git.resolve_ref(ctx.worker_branch)
            git.compare_and_swap_ref(target_ref, ctx.base_oid, worker_oid)
            git.remove_worktree(ctx.worktree_path, target=target_ref)
        except (GitOperationError, UnlandedWorkError):
            return MergeBackResult(rebased=False, worktree_preserved=str(ctx.worktree_path))
    return MergeBackResult(rebased=True, worktree_preserved=None)
