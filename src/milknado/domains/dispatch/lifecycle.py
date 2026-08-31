"""Sync-run dispatch and stale-node reclaim: the node-lifecycle domain operations."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

from milknado.domains.common import (
    GitPort,
    MikadoNode,
    NodeKind,
    NodeStatus,
    RunResult,
    WorktreeMode,
)
from milknado.domains.dispatch._runstate import make_run_id, now_iso, runs_dir
from milknado.domains.dispatch.brief import render_brief
from milknado.domains.dispatch.isolate import (
    IsolateContext,
    MergeBackResult,
    merge_back_isolated,
    setup_isolated_worktree,
)
from milknado.domains.dispatch.ports import (
    FinishDispatchPort,
    GraphLookupPort,
    ProcessPort,
    WorkerOutcomePort,
)
from milknado.domains.dispatch.reconcile import (
    fail_stale_running_runs,
    find_terminal_runs_for_node,
    latest_terminal_run,
    reconcile_node_status,
)
from milknado.domains.dispatch.runner import run_headless

if TYPE_CHECKING:
    from milknado.domains.graph import MikadoGraph, RunRecord

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SyncDispatchRequest:
    node_id: int
    project_root: Path
    worker_cmd: str | None
    timeout_seconds: int
    default_cmd: str
    process: ProcessPort
    brief_prepend: str | None = None
    worktree_mode: WorktreeMode = WorktreeMode.ISOLATE
    merge_back: bool = True
    worktree_pattern: str = ""


def _setup_sync_worktree(
    graph: MikadoGraph,
    git: GitPort,
    node: MikadoNode,
    run_id: str,
    request: SyncDispatchRequest,
) -> tuple[Path, IsolateContext | None]:
    if request.worktree_mode != WorktreeMode.ISOLATE:
        return request.project_root, None
    context = setup_isolated_worktree(
        graph,
        git,
        request.project_root,
        node,
        run_id,
        request.worktree_pattern,
    )
    return context.worktree_path, context


def _finish_dispatch(
    graph: object,
    node_id: int,
    run_id: str,
    result: object,
    terminal: str,
    merge: MergeBackResult | None,
) -> None:
    typed_graph = cast(FinishDispatchPort, graph)
    worker_result = cast(WorkerOutcomePort, result)
    if (
        typed_graph.finish_run(
            run_id,
            RunResult(
                status=terminal,
                exit_code=worker_result.exit_code,
                timed_out=worker_result.timed_out,
                ended_at=now_iso(),
                rebased=merge.rebased if merge is not None else None,
            ),
        )
        is False
    ):
        raise RuntimeError(f"terminal run write lost its fence for {run_id}")
    if (
        typed_graph.mark_terminal(
            node_id, run_id, NodeStatus.DONE if terminal == "done" else NodeStatus.FAILED
        )
        is False
    ):
        if typed_graph.get_node(node_id) is None:
            return
        raise RuntimeError(f"terminal node write lost its fence for node {node_id}")


def dispatch_node_sync(
    graph: GraphLookupPort,
    git: GitPort,
    request: SyncDispatchRequest,
) -> dict[str, object]:
    node = graph.get_node(request.node_id)
    if node is None:
        raise ValueError(f"node {request.node_id} not found")
    node = cast(MikadoNode, node)
    typed_graph = cast("MikadoGraph", graph)
    if node.kind != NodeKind.TASK:
        raise ValueError(
            f"node {request.node_id} has kind={node.kind.value}; only task nodes can be dispatched"
        )
    run_id = make_run_id(request.node_id)
    typed_graph.claim_node_for_dispatch(request.node_id, run_id, now=now_iso())
    log_path = runs_dir(request.project_root) / f"{run_id}.log"
    started = False
    try:
        brief = render_brief(
            typed_graph,
            request.node_id,
            prepend=request.brief_prepend,
            project_root=request.project_root,
        )
        cwd, isolate = _setup_sync_worktree(typed_graph, git, node, run_id, request)
        typed_graph.start_run(
            run_id,
            request.node_id,
            str(log_path),
            now_iso(),
            request.timeout_seconds,
        )
        started = True
        result = run_headless(
            request.project_root,
            request.node_id,
            brief,
            request.worker_cmd,
            request.timeout_seconds,
            run_id=run_id,
            default_cmd=request.default_cmd,
            cwd=cwd,
            process=request.process,
        )
        terminal = "done" if result.exit_code == 0 and not result.timed_out else "failed"
        merge = _maybe_merge_back(git, request, isolate, terminal)
        if merge is not None and not merge.rebased:
            terminal = "failed"
        _finish_dispatch(typed_graph, request.node_id, run_id, result, terminal, merge)
    except Exception as exc:
        terminal_error: BaseException | None = None
        try:
            run_written = True
            if started:
                run_written = typed_graph.finish_run(
                    run_id,
                    RunResult(
                        status="failed",
                        exit_code=-1,
                        timed_out=False,
                        ended_at=now_iso(),
                        error=f"{type(exc).__name__}: {exc}",
                    ),
                )
            node_written = typed_graph.mark_terminal(request.node_id, run_id, NodeStatus.FAILED)
            if run_written is False or node_written is False:
                terminal_error = RuntimeError(
                    f"terminal persistence lost its fence for run {run_id}"
                )
        except Exception as persist_exc:
            terminal_error = persist_exc
        if terminal_error is not None:
            raise terminal_error from exc
        raise
    final = typed_graph.get_node(request.node_id)
    if final is None:
        raise RuntimeError(f"node {request.node_id} not found after run completed")
    return {
        "run_id": run_id,
        "node_id": request.node_id,
        "status": final.status.value,
        "exit_code": result.exit_code,
        "timed_out": result.timed_out,
        "rebased": merge.rebased if merge is not None else None,
        "log_path": str(result.log_path),
        "summary": result.summary,
        "worktree_preserved": (merge.worktree_preserved if merge is not None else None),
    }


def _maybe_merge_back(
    git: GitPort,
    request: SyncDispatchRequest,
    context: IsolateContext | None,
    worker_terminal: str,
) -> MergeBackResult | None:
    if context is None or not request.merge_back or worker_terminal != "done":
        return None
    result = merge_back_isolated(git, request.project_root, context)
    if result.worktree_preserved is not None:
        _logger.warning(
            "ISOLATE merge-back for branch %s did not tear down; preserved worktree %s",
            context.worker_branch,
            result.worktree_preserved,
        )
    return result


def reclaim_stale_node(graph: MikadoGraph, node_id: int, fence_run_id: str | None) -> None:
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
    _ = fail_stale_running_runs(graph, node_id)
    # Filter terminal files to this node's fence BEFORE picking the latest, so a
    # stale run with a later ended_at cannot mask the current owner's file.
    winner: RunRecord | None = latest_terminal_run(
        find_terminal_runs_for_node(graph, node_id, run_id=fence_run_id)
    )
    if winner is not None:
        reconcile_node_status(graph, node_id, winner["status"], run_id=winner.get("run_id"))
