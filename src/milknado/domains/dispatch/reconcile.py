"""Run-state and node-status reconciliation helpers shared by the MCP dispatch tools."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from milknado.domains.common import NodeStatus, pid_alive
from milknado.domains.dispatch._runstate import now_iso as _now_iso

_logger = logging.getLogger(__name__)

# Grace beyond a run's own timeout before a still-"running" state file is
# treated as orphaned by a vanished worker thread (e.g. the server crashed).
_STALE_GRACE_SECONDS = 30


def fail_stale_running_runs(graph, node_id: int) -> list[dict]:  # noqa: ANN001
    """Flip this node's orphaned "running" runs to "failed".

    The async worker runs in a daemon thread; if the server process dies
    mid-run the thread dies with it and never writes a terminal state, leaving
    the run stuck on "running" and the node locked on RUNNING forever
    (orphan reconciliation only recovers runs that reached done/failed). For the
    worker model a run still "running" past its own timeout plus a grace margin
    can only mean the thread vanished — `_execute` kills and reaps the subprocess
    at timeout, so a live worker always reaches a terminal write. A detached-ralph
    runner has no such killer (its only timeout is `wait_for_next_completion`), so
    this also skips any run whose recorded pid is still alive — slow, not dead.
    Mark the rest failed so the caller's reconciliation can release the node.
    Returns the runs it flipped.
    """
    now = datetime.now(UTC)
    flipped: list[dict] = []
    for state in graph.runs_for_node(node_id):
        if state.get("status") != "running":
            continue
        started_at = state.get("started_at")
        timeout = state.get("timeout_seconds")
        if not started_at or timeout is None:
            continue
        try:
            started = datetime.fromisoformat(started_at)
        except ValueError:
            continue
        if (now - started).total_seconds() <= timeout + _STALE_GRACE_SECONDS:
            continue
        # A detached-ralph runner records its own pid; a live process past the
        # grace window is slow (e.g. wedged in `git rebase`, which has no
        # subprocess timeout), not orphaned. Flipping it would thrash a run about
        # to write "done" — and could cross-fail another run kind sharing the
        # node. The async-worker path records no pid, so its orphan sweep (server
        # crash → daemon thread vanished, no terminal write) is unchanged:
        # pid-unknown still falls through to the timeout flip.
        pid = state.get("pid")
        if pid is not None and pid_alive(pid):
            # Log the skip: a run wedged past timeout+grace that keeps getting
            # skipped as "slow, not dead" would otherwise leave no trace, since
            # the caller discards this function's return value.
            _logger.info(
                "stale-run sweep skipped live run %s (node %s, pid %s) — slow, not orphaned",
                state.get("run_id"),
                node_id,
                pid,
            )
            continue
        # A pid-recording detached run that reached here has a dead pid (the
        # liveness skip above did not fire), so "vanished" understates it — name
        # the exited process. The async-worker path records no pid and keeps the
        # daemon-thread "vanished" wording.
        error = (
            f"detached runner (pid {pid}) exited before writing terminal state (stale running run)"
            if pid is not None
            else "worker vanished before writing terminal state (stale running run)"
        )
        ended_at = _now_iso()
        graph.finish_run(
            state["run_id"],
            status="failed",
            exit_code=-1,
            timed_out=False,
            ended_at=ended_at,
            error=error,
        )
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


def reconcile_orphan_node(graph, node_id: int) -> None:  # noqa: ANN001
    """Flip stale 'running' runs to failed, reconcile the node's status to the
    latest terminal run. No-op if the node is not RUNNING.

    Shared entry-point for mcp_run (#39) and mcp_ralph (#54) — the three-call
    orphan-release pattern must not be duplicated inline in either caller.
    """
    fail_stale_running_runs(graph, node_id)
    winner = latest_terminal_run(find_terminal_runs_for_node(graph, node_id))
    if winner is not None:
        reconcile_node_status(graph, node_id, winner["status"])
