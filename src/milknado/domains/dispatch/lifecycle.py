"""Sync-run dispatch and stale-node reclaim: the node-lifecycle domain operations."""

from __future__ import annotations

from pathlib import Path

from milknado.domains.common import NodeStatus, WorktreeMode
from milknado.domains.dispatch._runstate import make_run_id, now_iso, runs_dir
from milknado.domains.dispatch.brief import render_brief
from milknado.domains.dispatch.isolate import (
    IsolateContext,
    MergeBackResult,
    merge_back_isolated,
    resolve_feature_branch,
    setup_isolated_worktree,
)
from milknado.domains.dispatch.reconcile import (
    fail_stale_running_runs,
    find_terminal_runs_for_node,
    latest_terminal_run,
    reconcile_node_status,
)
from milknado.domains.dispatch.runner import run_headless


def _ensure_running(graph, node_id: int) -> bool:  # noqa: ANN001
    """Move a node to RUNNING so a (re)run's terminal transition (RUNNING ->
    DONE/FAILED) is always valid. BLOCKED/FAILED are reset through PENDING
    first. A DONE node cannot be reopened by the state machine and is left
    untouched — its prior result stands and callers must not record a new
    terminal status for it. Returns True when the node is RUNNING afterward.
    """
    node = graph.get_node(node_id)
    if node is None or node.status == NodeStatus.DONE:
        return False
    if node.status == NodeStatus.RUNNING:
        return True
    if node.status in (NodeStatus.BLOCKED, NodeStatus.FAILED):
        graph.mark_pending(node_id)
    graph.mark_running(node_id)
    return True


def _setup_sync_worktree(
    graph,  # noqa: ANN001
    project_root: Path,
    node,  # noqa: ANN001
    run_id: str,
    worktree_mode: WorktreeMode,
    worktree_pattern: str,
) -> tuple[Path, str | None]:
    """Resolve the worker cwd for a running sync dispatch.

    ISOLATE: create the node's worktree + branch, record it on the (already
    run_id-fenced) node, and run there. THIS_BRANCH: run in the shared checkout.
    Returns ``(worker_cwd, worker_branch)`` — ``worker_branch`` is None under
    THIS_BRANCH (nothing to merge back).
    """
    if worktree_mode != WorktreeMode.ISOLATE:
        return project_root, None
    return setup_isolated_worktree(graph, project_root, node, run_id, worktree_pattern)


def dispatch_node_sync(
    graph,  # noqa: ANN001
    node_id: int,
    project_root: Path,
    worker_cmd: str | None,
    timeout_seconds: int,
    *,
    default_cmd: str,
    brief_prepend: str | None = None,
    worktree_mode: WorktreeMode = WorktreeMode.ISOLATE,
    merge_back: bool = True,
    worktree_pattern: str = "",
) -> dict:
    node = graph.get_node(node_id)
    if node is None:
        raise ValueError(f"node {node_id} not found")
    brief = render_brief(graph, node_id, prepend=brief_prepend)
    running = _ensure_running(graph, node_id)
    run_id = make_run_id(node_id)
    started_at = now_iso()
    log_path = runs_dir(project_root) / f"{run_id}.log"
    worker_cwd: Path = project_root
    worker_branch: str | None = None
    if running:
        # Fence the node on this run_id BEFORE set_worktree (which CAS-updates on
        # run_id), then insert a rescuable "running" run row BEFORE blocking in
        # run_headless: a client timeout + server kill mid-run would otherwise
        # strand the node RUNNING until fail_stale_running_runs releases it.
        graph.set_run_id(node_id, run_id)
        worker_cwd, worker_branch = _setup_sync_worktree(
            graph, project_root, node, run_id, worktree_mode, worktree_pattern
        )
        graph.start_run(run_id, node_id, str(log_path), started_at, timeout_seconds)
    # Only inject MILKNADO_RUN_ID when a 'running' run row exists. On a DONE-node
    # re-run no row is inserted (start_run is gated on `running`), so passing the
    # run_id would have the worker deposit into a row that doesn't exist —
    # milknado_deposit_result raises "run not found". Pass None instead.
    result = run_headless(
        project_root,
        node_id,
        brief,
        worker_cmd,
        timeout_seconds,
        run_id=run_id if running else None,
        default_cmd=default_cmd,
        cwd=worker_cwd,
    )
    worker_terminal = "done" if result.exit_code == 0 and not result.timed_out else "failed"
    merge = _maybe_merge_back(
        project_root, node, worker_branch, worker_cwd, merge_back, worker_terminal
    )
    if merge is not None and not merge.rebased:
        worker_terminal = "failed"
    if running:
        if worker_terminal == "done":
            graph.mark_done(node_id)
        else:
            graph.mark_failed(node_id)
    final = graph.get_node(node_id)
    if final is None:
        raise RuntimeError(f"node {node_id} not found after run completed")
    rebased = merge.rebased if merge is not None else None
    if running:
        graph.finish_run(
            run_id,
            status=worker_terminal,
            exit_code=result.exit_code,
            timed_out=result.timed_out,
            ended_at=now_iso(),
            rebased=rebased,
        )
    graph.set_run_id(node_id, run_id)
    return {
        "run_id": run_id,
        "node_id": node_id,
        "status": final.status.value,
        "exit_code": result.exit_code,
        "timed_out": result.timed_out,
        "rebased": rebased,
        "log_path": str(result.log_path),
        "summary": result.summary,
        "worktree_preserved": merge.worktree_preserved if merge is not None else None,
    }


def _maybe_merge_back(
    project_root: Path,
    node,  # noqa: ANN001
    worker_branch: str | None,
    worktree_path: Path,
    merge_back: bool,
    worker_terminal: str,
) -> MergeBackResult | None:
    """Rebase-merge an ISOLATE worktree back on a clean exit, else no-op.

    Returns None when there is nothing to merge (THIS_BRANCH, merge_back=False, or
    a non-``done`` worker) so the caller leaves ``rebased`` at None — matching the
    old shared-checkout dispatch that never rebased.
    """
    if worker_branch is None or not merge_back or worker_terminal != "done":
        return None
    ctx = IsolateContext(
        worktree_path=worktree_path,
        worker_branch=worker_branch,
        feature_branch=resolve_feature_branch(project_root),
        node_id=node.id,
        description=node.description,
    )
    return merge_back_isolated(project_root, ctx)


def reclaim_stale_node(graph, node_id: int, fence_run_id: str | None) -> None:  # noqa: ANN001
    """Reconcile an orphaned RUNNING node before a new async dispatch.

    A prior worker may have finished without anyone polling, leaving the node
    stuck on RUNNING. Also sweeps runs stuck on "running" past their timeout:
    the worker thread vanished (e.g. server crash) without writing a terminal
    state, which would lock the node just as permanently. Routes terminal writes
    through the atomic run_id + status fence (graph.mark_terminal), closing the
    TOCTOU a pre-check-then-unfenced-write left open: a node re-claimed under a
    newer run cannot be clobbered. A legacy node with no run_id has no fence to
    honour and falls back to the unconditional reconcile.
    """
    fail_stale_running_runs(graph, node_id)
    # Filter terminal files to this node's fence BEFORE picking the latest, so a
    # stale run with a later ended_at cannot mask the current owner's file.
    winner = latest_terminal_run(find_terminal_runs_for_node(graph, node_id, run_id=fence_run_id))
    if winner is not None:
        reconcile_node_status(graph, node_id, winner["status"], run_id=winner.get("run_id"))
