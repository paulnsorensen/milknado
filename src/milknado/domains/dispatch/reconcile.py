"""Run-state and node-status reconciliation helpers shared by the MCP dispatch tools."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from milknado.domains.common import NodeStatus, RunResult, pid_alive
from milknado.domains.dispatch._runstate import now_iso as _now_iso

_logger = logging.getLogger(__name__)

# Grace beyond a run's own timeout before a still-"running" state file is
# treated as orphaned by a vanished worker thread (e.g. the server crashed).
_STALE_GRACE_SECONDS = 30


def fail_stale_running_runs(graph, node_id: int) -> list[dict]:  # noqa: ANN001
    """Finalize confirmed-dead owners and timeout-stale pid-less runs.

    A run row's pid is its worker pid. In-process coordinator runs deliberately
    leave it NULL because cancelling that pid would kill the coordinator group;
    their durable, fenced node pid is used only as a liveness signal. Missing pid
    metadata never proves death and still follows the existing timeout path.
    """
    now = datetime.now(UTC)
    node = graph.get_node(node_id) if hasattr(graph, "get_node") else None
    flipped: list[dict] = []
    for state in graph.runs_for_node(node_id):
        if state.get("status") != "running":
            continue
        run_id = state["run_id"]
        worker_pid = state.get("pid")
        owner_pid = worker_pid
        if (
            owner_pid is None
            and node is not None
            and node.run_id == run_id
            and node.pid is not None
        ):
            owner_pid = node.pid
        if owner_pid is not None:
            if pid_alive(owner_pid):
                continue
            error = "worker session gone"
            ended_at = _now_iso()
            if (
                graph.finish_run(
                    run_id,
                    RunResult(
                        status="failed",
                        exit_code=-1,
                        timed_out=False,
                        ended_at=ended_at,
                        error=error,
                    ),
                )
                is False
            ):
                raise RuntimeError(f"stale terminal write lost its fence for {run_id}")
            flipped.append(
                {
                    **state,
                    "status": "failed",
                    "exit_code": -1,
                    "timed_out": False,
                    "ended_at": ended_at,
                    "error": error,
                }
            )
            continue
        started_at = state.get("started_at")
        timeout = state.get("timeout_seconds")
        if not started_at or timeout is None:
            continue
        try:
            started = datetime.fromisoformat(started_at)
        except ValueError:
            _logger.warning(
                "stale-run sweep skipped malformed timestamp: run_id=%s node_id=%d started_at=%r",
                run_id,
                node_id,
                started_at,
            )
            continue
        if (now - started).total_seconds() <= timeout + _STALE_GRACE_SECONDS:
            continue
        ended_at = _now_iso()
        error = "worker vanished before writing terminal state (stale running run)"
        if (
            graph.finish_run(
                run_id,
                RunResult(
                    status="failed",
                    exit_code=-1,
                    timed_out=False,
                    ended_at=ended_at,
                    error=error,
                ),
            )
            is False
        ):
            raise RuntimeError(f"stale terminal write lost its fence for {run_id}")
        flipped.append(
            {
                **state,
                "status": "failed",
                "exit_code": -1,
                "timed_out": False,
                "ended_at": ended_at,
                "error": error,
            }
        )
    return flipped


def reconcile_orphaned_runs(graph) -> list[dict]:  # noqa: ANN001
    """Finalize and release orphaned nodes before a new coordinator dispatches."""
    if not hasattr(graph, "get_all_nodes"):
        return []
    reconciled: list[dict] = []
    for node in graph.get_all_nodes():
        if node.status is not NodeStatus.RUNNING or node.run_id is None:
            continue
        fail_stale_running_runs(graph, node.id)
        terminal = latest_terminal_run(
            find_terminal_runs_for_node(graph, node.id, run_id=node.run_id)
        )
        if terminal is None:
            continue
        reconcile_node_status(graph, node.id, terminal["status"], run_id=node.run_id)
        reconciled.append(terminal)
    return reconciled

def find_terminal_runs_for_node(
    graph,
    node_id: int,
    run_id: str | None = None,  # noqa: ANN001
) -> list[dict]:
    """Return this node's runs that reached a terminal status. Used by callers that
    want to reconcile orphaned runs (started, worker finished, but never polled —
    node still marked RUNNING).

    When `run_id` is given, filter to that fence: only the node's current owner's
    terminal run is returned, so a stale terminal run from an older run cannot be
    selected. Callers must filter to the fence BEFORE picking the latest run —
    otherwise a stale run with a later `ended_at` could mask the owner's run."""
    return graph.runs_for_node(node_id, terminal_only=True, run_id=run_id)


def latest_terminal_run(runs: list[dict]) -> dict | None:
    """Return the most recent terminal run by ended_at, or None if runs is empty.

    Deterministic selection for reconcile: when a node has multiple terminal
    runs (e.g. a prior run finished after the node was reset and re-run), the
    latest ended_at wins. Callers that fence on run_id must filter to the fence
    first (see find_terminal_runs_for_node) so a stale run with a later ended_at
    cannot mask the current owner's terminal file. Shared by the headless
    (mcp_run) and detached-ralph (mcp_ralph) dispatch paths.
    """
    terminal = [r for r in runs if r.get("status") in ("done", "failed")]
    if not terminal:
        return None
    return max(terminal, key=lambda r: r.get("ended_at") or "")


def reconcile_node_status(graph, node_id: int, run_status: str, run_id: str | None = None) -> None:  # noqa: ANN001
    """Transition a node to its run's terminal status, idempotently and fenced.

    When `run_id` is given AND the node carries a run_id, the write goes through
    `graph.mark_terminal`, whose atomic `WHERE run_id = ? AND status = 'running'`
    is the real fence: a node re-claimed under a newer run_id (or already terminal)
    matches zero rows and is left for its current owner. This closes the TOCTOU a
    pre-check-then-unfenced-write left open — two concurrent reconcilers can both
    pass a Python-level run_id check, but only the matching atomic UPDATE lands.

    A node with no run_id (in-process TUI / legacy) has no fence to honour, so it
    falls back to the unconditional transition; only a RUNNING node is touched, so
    a node reset out of RUNNING is never force-marked. Shared by the sync (`mcp_run`)
    and detached worktree (`mcp_ralph`) dispatch tools.
    """
    _logger.info(
        "run reconciliation: run_id=%s node_id=%d status=%s",
        run_id,
        node_id,
        run_status,
    )
    node = graph.get_node(node_id)
    if node is None or node.status != NodeStatus.RUNNING:
        return
    target = {"done": NodeStatus.DONE, "failed": NodeStatus.FAILED}.get(run_status)
    if target is None:
        return
    if run_id is not None and node.run_id is not None:
        graph.mark_terminal(node_id, run_id, target)
    elif target is NodeStatus.DONE:
        graph.mark_done(node_id)
    else:
        graph.mark_failed(node_id)
