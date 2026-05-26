"""Reconcile stale RUNNING nodes after a crashed or interrupted run.

Called by `milknado run --resume` before the dispatch loop starts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from milknado.domains.common.protocols import GitPort, RalphPort
    from milknado.domains.graph import MikadoGraph

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReconcileResult:
    resolved_done: list[int] = field(default_factory=list)
    reset_pending: list[int] = field(default_factory=list)


def reconcile_stale_nodes(
    graph: MikadoGraph,
    git: GitPort,
    ralph: RalphPort,
    feature_branch: str,
) -> ReconcileResult:
    """Find every RUNNING node and resolve it against actual git state.

    Nodes whose work was merged into *feature_branch* are marked DONE.
    All others are reset to PENDING so the next dispatch loop picks them up.
    """
    from milknado.domains.common.types import NodeStatus

    running = [n for n in graph.get_all_nodes() if n.status == NodeStatus.RUNNING]
    if not running:
        return ReconcileResult()

    resolved_done: list[int] = []
    reset_pending: list[int] = []

    for node in running:
        _stop_run(ralph, node.run_id)
        if _appears_merged(node.branch_name, node.worktree_path, git, feature_branch):
            _logger.info("reconcile: node %d already merged — marking done", node.id)
            graph.mark_done(node.id)
            resolved_done.append(node.id)
        else:
            _clean_worktree(node.worktree_path, git)
            _logger.info("reconcile: node %d stale — resetting to pending", node.id)
            graph.mark_pending(node.id)
            reset_pending.append(node.id)

    return ReconcileResult(resolved_done=resolved_done, reset_pending=reset_pending)


def _stop_run(ralph: RalphPort, run_id: str | None) -> None:
    if not run_id:
        return
    try:
        if ralph.get_run(run_id) is not None:
            ralph.stop_run(run_id)
    except Exception as exc:
        _logger.warning("reconcile: stop_run(%s) failed: %s", run_id, exc)


def _appears_merged(
    branch_name: str | None,
    worktree_path: str | None,
    git: GitPort,
    feature_branch: str,
) -> bool:
    """True when the node branch was squash-rebased into feature_branch but mark_done
    never ran — i.e. rebase_and_merge completed but the process crashed before the DB
    was updated."""
    if not branch_name:
        return False
    # Worktree still present → rebase_and_merge never finished (it removes it).
    if worktree_path and Path(worktree_path).exists():
        return False
    try:
        return git.branch_exists(branch_name) and git.is_ancestor(branch_name, feature_branch)
    except Exception as exc:
        _logger.warning("reconcile: git check for branch %r failed: %s", branch_name, exc)
        return False


def _clean_worktree(worktree_path: str | None, git: GitPort) -> None:
    if not worktree_path:
        return
    wt = Path(worktree_path)
    if not wt.exists():
        return
    try:
        git.remove_worktree(wt)
    except Exception as exc:
        _logger.warning("reconcile: remove_worktree(%s) failed: %s", wt, exc)
