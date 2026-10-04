"""ISOLATE-mode worktree creation + merge-back for one-shot run_inline dispatch.

ISOLATE mode runs a worker in its own git worktree/branch. When ``merge_back``
is enabled, this module performs the direct GitPort merge-back on exit 0:
squash, rebase, compare-and-swap the captured target ref, and remove the
worktree only after the landing succeeds. Failed or mismatched operations
preserve the worktree for inspection.
"""

from __future__ import annotations

import logging
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from filelock import FileLock

from milknado.domains.common import GitPort, MikadoNode, UnlandedWorkError, slugify
from milknado.domains.common.errors import GitOperationError
from milknado.domains.graph import ExecutionGroup, ExecutionGroupStore, GroupWorkspace

if TYPE_CHECKING:
    from milknado.domains.graph import MikadoGraph

_logger = logging.getLogger(__name__)


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


@dataclass(frozen=True)
class GroupWorktreeRequest:
    graph_id: str
    task_ids: tuple[int, ...]
    provider_session_id: str
    description: str
    source_group_id: str | None = None


def setup_group_worktree(
    store: ExecutionGroupStore,
    git: GitPort,
    root: Path,
    request: GroupWorktreeRequest,
) -> ExecutionGroup:
    """Create a distinct checkout and persist its group owner."""
    if request.source_group_id is not None and request.task_ids:
        raise ValueError("fork copies source tasks; task_ids must be empty")
    if request.source_group_id is not None:
        source = store.get(request.source_group_id)
        if source is None:
            raise ValueError("source execution group does not exist")
        if git.current_branch() != source.branch_name:
            raise ValueError("fork must start from the source group branch")
    token = uuid4().hex[:12]
    slug = slugify(request.description, max_length=30) or "group"
    path = root.parent / f"{root.name}-group-{token}-{slug}"
    branch = f"milknado/group-{token}-{slug}"
    base_oid = git.resolve_ref(f"refs/heads/{git.current_branch()}")
    _ = git.create_worktree(path, branch)
    try:
        if git.resolve_ref(branch) != base_oid:
            raise GitOperationError("checkout changed while group worktree was being created")
        workspace = GroupWorkspace(str(path), branch, request.provider_session_id)
        if request.source_group_id is None:
            return store.create(request.graph_id, request.task_ids, workspace)
        return store.fork(request.source_group_id, workspace)
    except Exception:
        discard_isolated_worktree(git, path, branch)
        raise


def _create_node_worktree(
    git: GitPort,
    root: Path,
    node_id: int,
    description: str,
    worktree_pattern: str,
) -> IsolateContext:
    slug = slugify(description, max_length=30) or "node"
    wt_path = root / worktree_pattern.format(node_id=node_id, slug=slug)
    if not wt_path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"worktree path resolves outside project_root: {wt_path!r}")
    target_branch = git.current_branch()
    target_ref = f"refs/heads/{target_branch}"
    base_oid = git.resolve_ref(target_ref)
    worker_branch = f"milknado/{node_id}-{slug}"
    _ = git.create_worktree(wt_path, worker_branch)
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


def discard_isolated_worktree(git: GitPort, worktree_path: Path, branch: str) -> None:
    """Remove a never-started worker checkout and its fresh branch so a retry can recreate both."""
    git.force_remove_worktree(worktree_path)
    git.delete_branch(branch)


@contextmanager
def _merge_back_lock(root: Path) -> Generator[None, None, None]:
    """Serialize merge-backs with a portable, process-scoped file lock."""
    lock_dir = root / ".milknado"
    lock_dir.mkdir(parents=True, exist_ok=True)
    with FileLock(lock_dir / "merge-back.lock", timeout=-1):
        yield


def setup_isolated_worktree(
    graph: MikadoGraph,
    git: GitPort,
    root: Path,
    node: MikadoNode,
    run_id: str,
    worktree_pattern: str,
) -> IsolateContext:
    context = create_isolated_worktree(git, root, node.id, node.description, worktree_pattern)
    try:
        graph.set_worktree(node.id, run_id, str(context.worktree_path), context.worker_branch)
    except Exception:
        git.force_remove_worktree(context.worktree_path)
        raise
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
            _ = git.squash_and_commit(
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


def merge_back_if_done(
    git: GitPort,
    root: Path,
    context: IsolateContext | None,
    terminal: str,
) -> MergeBackResult | None:
    if context is None or terminal != "done":
        return None
    result = merge_back_isolated(git, root, context)
    if result.worktree_preserved is not None:
        _logger.warning(
            "ISOLATE merge-back for branch %s did not tear down; preserved worktree %s",
            context.worker_branch,
            result.worktree_preserved,
        )
    return result
