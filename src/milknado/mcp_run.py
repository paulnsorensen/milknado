"""Milknado MCP worker-run tools — sync and async headless dispatch."""

from __future__ import annotations

from pathlib import Path

from milknado._mcp_core import mcp, open_graph, resolve_project_root
from milknado.domains.common import NodeStatus
from milknado.domains.dispatch import (
    fail_stale_running_runs,
    find_terminal_runs_for_node,
    poll_async_run,
    reconcile_node_status,
    render_brief,
    run_headless,
    start_headless_async,
)


def _ensure_running(graph, node_id: int) -> bool:
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


def _run_and_update_status(
    graph,
    node_id: int,
    project_root: Path,
    worker_cmd: str | None,
    timeout_seconds: int,
    *,
    brief_prepend: str | None = None,
) -> dict:
    node = graph.get_node(node_id)
    if node is None:
        raise ValueError(f"node {node_id} not found")
    brief = render_brief(graph, node_id, prepend=brief_prepend)
    running = _ensure_running(graph, node_id)
    result = run_headless(project_root, node_id, brief, worker_cmd, timeout_seconds)
    if running:
        if result.exit_code == 0 and not result.timed_out:
            graph.mark_done(node_id)
        else:
            graph.mark_failed(node_id)
    final = graph.get_node(node_id)
    assert final is not None
    return {
        "node_id": node_id,
        "status": final.status.value,
        "exit_code": result.exit_code,
        "timed_out": result.timed_out,
        "log_path": str(result.log_path),
        "summary": result.summary,
    }


@mcp.tool()
def milknado_todo_run(
    node_id: int,
    worker_cmd: str | None = None,
    timeout_seconds: int = 600,
    project_root: str = "",
) -> dict:
    """Spawn a subprocess worker with the task brief on stdin; capture log and update status.

    worker_cmd defaults to $MILKNADO_WORKER_CMD then `claude -p`.
    On exit 0 the node is marked done; on nonzero/timeout it is marked failed.
    """
    root = resolve_project_root(project_root or None)
    graph, cfg = open_graph(root)
    try:
        return _run_and_update_status(
            graph,
            node_id,
            root,
            worker_cmd,
            timeout_seconds,
            brief_prepend=cfg.worker_brief_prepend,
        )
    finally:
        graph.close()


@mcp.tool()
def milknado_todo_run_start(
    node_id: int,
    worker_cmd: str | None = None,
    timeout_seconds: int = 600,
    project_root: str = "",
) -> dict:
    """Start a worker asynchronously; returns immediately with a run_id for polling.

    Refuses if the node is already running. Use milknado_todo_run_poll(run_id) to
    check progress; node status is reconciled to done/failed on the first poll
    after the worker exits.
    """
    root = resolve_project_root(project_root or None)
    graph, cfg = open_graph(root)
    try:
        node = graph.get_node(node_id)
        if node is None:
            raise ValueError(f"node {node_id} not found")
        if node.status == NodeStatus.RUNNING:
            # Before refusing, reconcile any orphaned terminal runs for this
            # node — a prior worker may have finished without anyone polling,
            # leaving the node status stuck on RUNNING. Without this the node
            # is locked out forever. Also sweep runs stuck on "running" past
            # their timeout: the worker thread vanished (e.g. server crash)
            # without writing a terminal state, which would lock the node
            # just as permanently.
            fail_stale_running_runs(root, node_id)
            for state in find_terminal_runs_for_node(root, node_id):
                reconcile_node_status(graph, node_id, state["status"])
            node = graph.get_node(node_id)
            assert node is not None
            if node.status == NodeStatus.RUNNING:
                raise ValueError(
                    f"node {node_id} is already running; set status back to pending to retry"
                )
        brief = render_brief(graph, node_id, prepend=cfg.worker_brief_prepend)
        # Normalise to RUNNING before dispatch so the post-run poll
        # reconciliation (RUNNING -> DONE/FAILED) is always a valid transition,
        # even when re-running a node that was FAILED/BLOCKED from a prior run.
        _ensure_running(graph, node_id)
        ref = start_headless_async(root, node_id, brief, worker_cmd, timeout_seconds)
        return {
            "run_id": ref.run_id,
            "node_id": node_id,
            "status": "running",
            "log_path": str(ref.log_path),
            "state_path": str(ref.state_path),
        }
    finally:
        graph.close()


@mcp.tool()
def milknado_todo_run_poll(run_id: str, project_root: str = "") -> dict:
    """Poll an async run; reconciles node status to done/failed once the worker exits."""
    root = resolve_project_root(project_root or None)
    state = poll_async_run(root, run_id)
    if state["status"] in ("done", "failed"):
        graph, _cfg = open_graph(root)
        try:
            reconcile_node_status(graph, state["node_id"], state["status"])
        finally:
            graph.close()
    return state
