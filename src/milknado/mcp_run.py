"""Milknado MCP worker-run tools — sync and async headless dispatch."""

from __future__ import annotations

import json
import logging
import os
import shlex
import signal
from pathlib import Path

from milknado._mcp_core import mcp, open_graph, resolve_project_root
from milknado.adapters import GitAdapter
from milknado.domains.common import NodeStatus
from milknado.domains.dispatch import (
    make_run_id,
    now_iso,
    poll_async_run,
    read_state,
    reconcile_node_status,
    reconcile_orphan_node,
    render_brief,
    run_headless,
    runs_dir,
    start_headless_async,
    write_state,
)
from milknado.domains.dispatch._runstate import RUN_ID_RE
from milknado.domains.dispatch.runner import _dispatch_lock, _validate_worker_argv

_logger = logging.getLogger(__name__)


def _validate_worker_cmd(worker_cmd: str | None) -> None:
    """Reject an explicit worker_cmd whose executable isn't an allowed AI agent CLI.

    Eager pre-check on the MCP arg; the env fallback and built-in default are
    validated again where they're resolved (`runner._resolve_worker_cmd`). Both
    routes share `_validate_worker_argv`, so the allowlist lives in one place.
    """
    if not worker_cmd or not worker_cmd.strip():
        return
    _validate_worker_argv(shlex.split(worker_cmd))


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
    run_id = make_run_id(node_id)
    result = run_headless(project_root, node_id, brief, worker_cmd, timeout_seconds, run_id=run_id)
    worker_terminal = "done" if result.exit_code == 0 and not result.timed_out else "failed"
    if running:
        if worker_terminal == "done":
            graph.mark_done(node_id)
        else:
            graph.mark_failed(node_id)
    final = graph.get_node(node_id)
    if final is None:
        raise RuntimeError(f"node {node_id} not found after run completed")
    rdir = runs_dir(project_root)
    state_path = rdir / f"{run_id}.state.json"
    write_state(
        state_path,
        {
            "run_id": run_id,
            "node_id": node_id,
            "status": worker_terminal,
            "exit_code": result.exit_code,
            "timed_out": result.timed_out,
            "log_path": str(result.log_path),
            "rebased": None,
        },
    )
    graph.set_run_id(node_id, run_id)
    return {
        "run_id": run_id,
        "node_id": node_id,
        "status": final.status.value,
        "exit_code": result.exit_code,
        "timed_out": result.timed_out,
        "rebased": None,
        "log_path": str(result.log_path),
        "state_path": str(state_path),
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
    Blocks for up to timeout_seconds (default 600). For non-blocking dispatch
    use milknado_todo_run_start / milknado_todo_run_poll.
    """
    _validate_worker_cmd(worker_cmd)
    root = resolve_project_root(project_root or None)
    graph, cfg = open_graph(root)
    _logger.info("milknado_todo_run: node=%d timeout=%ds", node_id, timeout_seconds)
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
    _validate_worker_cmd(worker_cmd)
    root = resolve_project_root(project_root or None)
    graph, cfg = open_graph(root)
    try:
        with _dispatch_lock(root, node_id):
            node = graph.get_node(node_id)
            if node is None:
                raise ValueError(f"node {node_id} not found")
            if node.status == NodeStatus.RUNNING:
                reconcile_orphan_node(graph, root, node_id)
                node = graph.get_node(node_id)
                if node is None:
                    raise RuntimeError(f"node {node_id} not found after orphan reconcile")
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
            graph.set_run_id(node_id, ref.run_id)
            _logger.info(
                "milknado_todo_run_start: node=%d run_id=%s timeout=%ds",
                node_id,
                ref.run_id,
                timeout_seconds,
            )
        return {
            "run_id": ref.run_id,
            "node_id": node_id,
            "status": "running",
            "exit_code": None,
            "timed_out": None,
            "rebased": None,
            "log_path": str(ref.log_path),
            "state_path": str(ref.state_path),
            "summary": None,
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
    rdir = runs_dir(root)
    state["state_path"] = str(rdir / f"{run_id}.state.json")
    state.setdefault("rebased", None)
    return state


@mcp.tool()
def milknado_run_list(project_root: str = "") -> list[dict]:
    """List active and recent runs from .milknado/runs/, sorted newest first.

    Returns superset-schema dicts. summary is always None — use
    milknado_todo_run_poll or milknado_ralph_run_poll for log tails.
    """
    root = resolve_project_root(project_root or None)
    rdir = runs_dir(root)
    entries: list[tuple[str, dict]] = []
    for sp in rdir.glob("*.state.json"):
        try:
            state = read_state(sp)
        except (OSError, json.JSONDecodeError):
            continue
        entries.append(
            (
                state.get("started_at") or "",
                {
                    "run_id": state.get("run_id"),
                    "node_id": state.get("node_id"),
                    "status": state.get("status"),
                    "exit_code": state.get("exit_code"),
                    "timed_out": state.get("timed_out"),
                    "rebased": state.get("rebased"),
                    "log_path": state.get("log_path"),
                    "state_path": str(sp),
                    "summary": None,
                },
            )
        )
    entries.sort(key=lambda x: x[0], reverse=True)
    return [e[1] for e in entries]


def _state_to_run_dict(state: dict, state_path: Path) -> dict:
    return {
        "run_id": state.get("run_id"),
        "node_id": state.get("node_id"),
        "status": state.get("status"),
        "exit_code": state.get("exit_code"),
        "timed_out": state.get("timed_out"),
        "rebased": state.get("rebased"),
        "log_path": state.get("log_path"),
        "state_path": str(state_path),
        "summary": None,
    }


@mcp.tool()
def milknado_run_cancel(run_id: str, project_root: str = "") -> dict:
    """Cancel a run: signal the process group, reconcile node status, prune worktree.

    No-ops cleanly if the run is already terminal. Returns the final run state
    in the unified superset schema.
    """
    root = resolve_project_root(project_root or None)
    if not RUN_ID_RE.match(run_id):
        raise ValueError(f"invalid run_id format: {run_id!r}")
    rdir = runs_dir(root)
    state_path = rdir / f"{run_id}.state.json"
    if not state_path.exists():
        raise ValueError(f"run {run_id!r} not found")
    state = read_state(state_path)
    if state.get("status") != "running":
        return _state_to_run_dict(state, state_path)

    pid = state.get("pid")
    if pid is not None:
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except (ProcessLookupError, OSError):
            pass  # process already gone

    cancelled = {
        **state,
        "status": "failed",
        "exit_code": -1,
        "timed_out": False,
        "ended_at": now_iso(),
        "error": "cancelled",
    }
    write_state(state_path, cancelled)

    node_id = state.get("node_id")
    if node_id is not None:
        graph, _cfg = open_graph(root)
        try:
            node = graph.get_node(node_id)
            if node is not None and node.worktree_path:
                wt = Path(node.worktree_path)
                if wt.exists():
                    try:
                        GitAdapter(root).remove_worktree(wt)
                    except Exception as exc:
                        _logger.warning("Failed to remove worktree %s on cancel: %s", wt, exc)
            reconcile_node_status(graph, node_id, "failed")
        finally:
            graph.close()
        try:
            GitAdapter(root).prune_worktrees()
        except Exception as exc:
            _logger.warning("git worktree prune failed on cancel: %s", exc)

    _logger.info("milknado_run_cancel: run_id=%s node_id=%s", run_id, node_id)
    return _state_to_run_dict(cancelled, state_path)
