"""Milknado MCP todo + dispatch tools — registered against the shared FastMCP instance."""

from __future__ import annotations

from pathlib import Path

from milknado._mcp_core import mcp, open_graph, resolve_project_root
from milknado.domains.common import MikadoNode, NodeKind, NodeStatus
from milknado.domains.dispatch import (
    fail_stale_running_runs,
    find_terminal_runs_for_node,
    poll_async_run,
    render_brief,
    run_headless,
    start_headless_async,
)

_TODO_STATUS_MAP = {
    "pending": NodeStatus.PENDING,
    "in_progress": NodeStatus.RUNNING,
    "blocked": NodeStatus.BLOCKED,
    "done": NodeStatus.DONE,
}


def _parse_kind(value: str) -> NodeKind:
    try:
        return NodeKind(value)
    except ValueError as exc:
        valid = sorted(k.value for k in NodeKind)
        raise ValueError(f"invalid kind {value!r}; expected one of {valid}") from exc


def _parse_todo_status(value: str) -> NodeStatus:
    if value not in _TODO_STATUS_MAP:
        raise ValueError(f"invalid status {value!r}; expected one of {sorted(_TODO_STATUS_MAP)}")
    return _TODO_STATUS_MAP[value]


def _node_to_summary(node: MikadoNode) -> dict:
    return {
        "id": node.id,
        "kind": node.kind.value,
        "status": node.status.value,
        "description": node.description,
    }


def _build_subtree(node: MikadoNode, children_map: dict[int, list[MikadoNode]]) -> dict:
    payload = _node_to_summary(node)
    payload["children"] = [_build_subtree(c, children_map) for c in children_map.get(node.id, [])]
    return payload


def _apply_todo_status(graph, node: MikadoNode, target: NodeStatus) -> None:
    """Move a node to one of the todo facade's statuses.

    The facade exposes pending/in_progress/blocked/done, but the underlying
    state machine only allows BLOCKED -> PENDING and forbids PENDING -> DONE
    directly. Step through PENDING/RUNNING as needed so every reachable facade
    status is reachable. Setting a node to its current status is a no-op.
    Reopening a DONE node is intentionally disallowed and still raises
    InvalidTransition.
    """
    current = node.status
    if current == target:
        return
    if current == NodeStatus.BLOCKED and target in (NodeStatus.RUNNING, NodeStatus.DONE):
        graph.mark_pending(node.id)
        current = NodeStatus.PENDING
    if target == NodeStatus.DONE and current == NodeStatus.PENDING:
        graph.mark_running(node.id)
    dispatch = {
        NodeStatus.PENDING: graph.mark_pending,
        NodeStatus.RUNNING: graph.mark_running,
        NodeStatus.DONE: graph.mark_done,
        NodeStatus.BLOCKED: graph.mark_blocked,
    }
    dispatch[target](node.id)


@mcp.tool()
def milknado_todo_tree(project_root: str = "", root_id: int | None = None) -> list[dict]:
    """Return the todo tree from root_id, or the forest of all top-level nodes."""
    root = resolve_project_root(project_root or None)
    graph, _cfg = open_graph(root)
    try:
        children_map = graph.get_children_map()
        if root_id is not None:
            node = graph.get_node(root_id)
            if node is None:
                raise ValueError(f"node {root_id} not found")
            return [_build_subtree(node, children_map)]
        return [_build_subtree(n, children_map) for n in graph.get_roots()]
    finally:
        graph.close()


@mcp.tool()
def milknado_todo_add(
    description: str,
    kind: str = "task",
    parent_id: int | None = None,
    project_root: str = "",
) -> dict:
    """Add a todo node (kind: roadmap|goal|task), optionally linked under parent_id."""
    node_kind = _parse_kind(kind)
    root = resolve_project_root(project_root or None)
    graph, _cfg = open_graph(root)
    try:
        node = graph.add_node(description, parent_id=parent_id, kind=node_kind)
        return _node_to_summary(node)
    finally:
        graph.close()


@mcp.tool()
def milknado_todo_set_status(node_id: int, status: str, project_root: str = "") -> dict:
    """Set todo status: pending | in_progress | blocked | done."""
    target = _parse_todo_status(status)
    root = resolve_project_root(project_root or None)
    graph, _cfg = open_graph(root)
    try:
        node = graph.get_node(node_id)
        if node is None:
            raise ValueError(f"node {node_id} not found")
        _apply_todo_status(graph, node, target)
        updated = graph.get_node(node_id)
        assert updated is not None
        return _node_to_summary(updated)
    finally:
        graph.close()


@mcp.tool()
def milknado_todo_next(kind: str = "task", project_root: str = "") -> dict | None:
    """Return the next runnable node (leaf with no incomplete prereqs)."""
    node_kind = _parse_kind(kind)
    root = resolve_project_root(project_root or None)
    graph, _cfg = open_graph(root)
    try:
        node = graph.get_next_runnable(node_kind)
        return _node_to_summary(node) if node else None
    finally:
        graph.close()


@mcp.tool()
def milknado_todo_brief(node_id: int, project_root: str = "") -> dict:
    """Render a markdown brief for a task (description, ancestor goals, prereqs, files)."""
    root = resolve_project_root(project_root or None)
    graph, _cfg = open_graph(root)
    try:
        brief = render_brief(graph, node_id)
        files = graph.get_file_ownership(node_id)
        return {"node_id": node_id, "brief": brief, "files": files}
    finally:
        graph.close()


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
) -> dict:
    node = graph.get_node(node_id)
    if node is None:
        raise ValueError(f"node {node_id} not found")
    brief = render_brief(graph, node_id)
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
    graph, _cfg = open_graph(root)
    try:
        return _run_and_update_status(graph, node_id, root, worker_cmd, timeout_seconds)
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
    graph, _cfg = open_graph(root)
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
                _reconcile_node_status(graph, node_id, state["status"])
            node = graph.get_node(node_id)
            assert node is not None
            if node.status == NodeStatus.RUNNING:
                raise ValueError(
                    f"node {node_id} is already running; set status back to pending to retry"
                )
        brief = render_brief(graph, node_id)
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


def _reconcile_node_status(graph, node_id: int, run_status: str) -> None:
    node = graph.get_node(node_id)
    # Only a RUNNING node can validly transition to a terminal status. Any other
    # state (already terminal, or reset externally) is left alone so
    # reconciliation never raises InvalidTransition — the first reconcile of a
    # run wins, and a node taken out of RUNNING is not force-marked.
    if node is None or node.status != NodeStatus.RUNNING:
        return
    if run_status == "done":
        graph.mark_done(node_id)
    elif run_status == "failed":
        graph.mark_failed(node_id)


@mcp.tool()
def milknado_todo_run_poll(run_id: str, project_root: str = "") -> dict:
    """Poll an async run; reconciles node status to done/failed once the worker exits."""
    root = resolve_project_root(project_root or None)
    state = poll_async_run(root, run_id)
    if state["status"] in ("done", "failed"):
        graph, _cfg = open_graph(root)
        try:
            _reconcile_node_status(graph, state["node_id"], state["status"])
        finally:
            graph.close()
    return state
