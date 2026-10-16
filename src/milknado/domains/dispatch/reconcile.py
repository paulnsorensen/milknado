"""Run-state and node-status reconciliation helpers shared by the MCP dispatch tools."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast

from milknado.domains.common import MikadoNode, NodeStatus, RunResult, pid_alive
from milknado.domains.dispatch._attempt_completion import (
    AttemptRef,
    settle_cancelled,
    settle_counted_failure,
    settle_node_terminal,
    settle_preserved,
)
from milknado.domains.dispatch._runstate import exec_receipt_path
from milknado.domains.dispatch._runstate import now_iso as _now_iso
from milknado.domains.dispatch.ports import ProcessTerminationPort, StaleRunPort
from milknado.domains.dispatch.reap import ReapRequest, reap_orphaned_workers
from milknado.domains.dispatch.tmux_run import read_exec_evidence
from milknado.domains.graph import FinalizeTaskFailure, MikadoGraph, NodeWorkers

if TYPE_CHECKING:
    from milknado.domains.graph import RunRecord

_logger = logging.getLogger(__name__)

# Grace beyond a run's own timeout before a still-"running" state file is
# treated as orphaned by a vanished worker thread (e.g. the server crashed).
_STALE_GRACE_SECONDS = 30


def _observe_tmux_receipt(graph: StaleRunPort, state: dict[str, object]) -> None:
    if not isinstance(graph, MikadoGraph):
        return
    run_id = state.get("run_id")
    attempt_id = state.get("attempt_id")
    if not isinstance(run_id, str) or not isinstance(attempt_id, str):
        return
    nonce = graph.runs.latest_message(run_id, "tmux_exec_nonce")
    if nonce is None:
        return
    receipt = exec_receipt_path(graph.db_path.parent, run_id, nonce)
    evidence = read_exec_evidence(receipt, nonce)
    attempt = graph.task_attempts.get(attempt_id)
    if attempt is None:
        raise ValueError(f"task attempt {attempt_id} missing during tmux recovery")
    if evidence == "executed":
        _ = graph.task_attempts.confirm_execution(attempt_id, attempt.owner_run_id, run_id)
    elif evidence == "startup_failed":
        _ = graph.task_attempts.record_startup_failure(attempt_id, attempt.owner_run_id, run_id)


def _canonical_result(outcome: str, result: RunResult) -> RunResult:
    if outcome != "cancelled":
        return result
    return replace(result, error="cancelled", detail=None)


def _bind_group_failure(graph: MikadoGraph, request: FinalizeTaskFailure) -> FinalizeTaskFailure:
    """Bind a native group member's failure to its reservation so the fence resolves it."""
    group = graph.groups.for_task(request.node_id)
    if group is None:
        return request
    result = request.result
    return replace(
        request, group_id=group.id, group_result=result.detail or result.error or "failed"
    )


def _finish_stale_run(
    graph: StaleRunPort, state: dict[str, object], result: RunResult
) -> RunResult | None:
    run_id = state["run_id"]
    assert isinstance(run_id, str)
    attempt_id = state.get("attempt_id")
    if isinstance(graph, MikadoGraph) and isinstance(attempt_id, str):
        attempt = graph.task_attempts.get(attempt_id)
        if attempt is None:
            raise ValueError(f"task attempt {attempt_id} missing during stale recovery")
        if attempt.outcome != "active":
            canonical = _canonical_result(attempt.outcome, result)
            graph.runs.finish(run_id, canonical)
            return canonical
        disposition = graph.task_attempts.fail(
            _bind_group_failure(
                graph,
                FinalizeTaskFailure(
                    attempt.node_id, attempt_id, attempt.owner_run_id, run_id, result
                ),
            )
        )
        status = disposition.status
        return result if isinstance(status, NodeStatus) or status == "startup_failed" else None
    graph.runs.finish(run_id, result)
    return result


def fail_stale_running_runs(
    graph: StaleRunPort, node_id: int, process: ProcessTerminationPort
) -> list[dict[str, object]]:
    """Finalize confirmed-dead owners and timeout-stale pid-less runs."""
    now = datetime.now(UTC)
    graph_with_node = getattr(graph, "get_node", None)
    node = cast(MikadoNode | None, graph_with_node(node_id)) if callable(graph_with_node) else None
    flipped: list[dict[str, object]] = []
    for state in graph.runs.for_node(node_id):
        if state.get("status") != "running":
            continue
        run_id = state.get("run_id")
        if not isinstance(run_id, str):
            continue
        worker_pid = state.get("pid")
        owner_pid = (
            worker_pid
            if isinstance(worker_pid, int)
            else node.pid
            if node is not None and node.run_id == run_id
            else None
        )
        if owner_pid is not None:
            if pid_alive(owner_pid):
                continue
            error = "worker session gone"
        else:
            started_at = state.get("started_at")
            timeout = state.get("timeout_seconds")
            if not isinstance(started_at, str) or not isinstance(timeout, int):
                continue
            try:
                started = datetime.fromisoformat(started_at)
            except ValueError:
                _logger.warning(
                    "stale-run sweep skipped malformed timestamp: "
                    + "run_id=%s node_id=%d started_at=%r",
                    run_id,
                    node_id,
                    started_at,
                )
                continue
            if (now - started).total_seconds() <= timeout + _STALE_GRACE_SECONDS:
                continue
            error = "worker vanished before writing terminal state (stale running run)"
        _observe_tmux_receipt(graph, dict(state))
        if not reap_orphaned_workers(graph, process, ReapRequest(NodeWorkers(node_id))):
            continue
        result = RunResult(
            status="failed",
            exit_code=-1,
            timed_out=False,
            ended_at=_now_iso(),
            error=error,
        )
        finished = _finish_stale_run(graph, dict(state), result)
        if finished is None:
            continue
        flipped.append(
            {
                **state,
                "status": "failed",
                "exit_code": -1,
                "timed_out": False,
                "ended_at": finished.ended_at,
                "error": finished.error,
            }
        )
    return flipped


def _reconcile_node(
    graph: MikadoGraph, node: MikadoNode, process: ProcessTerminationPort
) -> RunRecord | None:
    assert node.run_id is not None
    _ = fail_stale_running_runs(cast(StaleRunPort, graph), node.id, process)
    terminal = graph.runs.latest_terminal(node.id, node.run_id)
    if terminal is None or not reap_orphaned_workers(
        graph, process, ReapRequest(NodeWorkers(node.id))
    ):
        return None
    reconcile_node_status(graph, node.id, terminal["status"], run_id=node.run_id)
    return terminal


def reconcile_orphaned_runs(graph: object, process: ProcessTerminationPort) -> list[RunRecord]:
    """Finalize and release orphaned nodes before a new coordinator dispatches."""
    get_all_nodes = getattr(graph, "get_all_nodes", None)
    if not callable(get_all_nodes):
        return []
    typed_graph = cast("MikadoGraph", graph)
    reconciled: list[RunRecord] = []
    nodes = cast(Iterable[object], get_all_nodes())
    for node_value in nodes:
        node = cast(MikadoNode, node_value)
        if node.status is not NodeStatus.RUNNING or node.run_id is None:
            continue
        try:
            terminal = _reconcile_node(typed_graph, node, process)
        except Exception:
            # Leave the node unreconciled so the next sweep retries it.
            _logger.exception("reconcile failed for node %s run %s", node.id, node.run_id)
            continue
        if terminal is not None:
            reconciled.append(terminal)
    return reconciled


def _is_preserved_run(row: RunRecord) -> bool:
    """A clean worker exit whose merge-back did not rebase is a preserved attempt."""
    return row["rebased"] is False and row["exit_code"] == 0 and row["error"] is None


def _run_result_from_row(row: RunRecord) -> RunResult:
    return RunResult(
        status="failed",
        exit_code=row["exit_code"],
        timed_out=bool(row["timed_out"]),
        ended_at=row["ended_at"] or _now_iso(),
        error=row["error"],
        detail=row["detail"],
    )


def _reconcile_failed_attempt(
    graph: MikadoGraph, node: MikadoNode, run_id: str, row: RunRecord
) -> None:
    """Settle an attempt-linked FAILED run: cancelled, preserved, startup-failed, or counted."""
    attempt_id = row["attempt_id"]
    assert isinstance(attempt_id, str) and node.run_id is not None
    attempt = graph.task_attempts.get(attempt_id)
    if attempt is None:
        raise ValueError(f"task attempt {attempt_id} missing during recovery")
    ref = AttemptRef(node.id, attempt_id, node.run_id, run_id)
    if row["error"] == "cancelled":
        settle_cancelled(graph, ref)
    elif attempt.outcome == "preserved" or (
        attempt.outcome == "active" and _is_preserved_run(row)
    ):
        owned = replace(ref, owner_run_id=attempt.owner_run_id)
        settle_preserved(graph, owned, record_outcome=attempt.outcome == "active")
    elif attempt.outcome == "startup_failed":
        _ = settle_node_terminal(graph, ref, NodeStatus.FAILED)
    elif attempt.outcome == "active":
        request = FinalizeTaskFailure(
            node.id, attempt_id, node.run_id, run_id, _run_result_from_row(row)
        )
        _ = settle_counted_failure(graph, _bind_group_failure(graph, request))


def reconcile_node_status(
    graph: MikadoGraph, node_id: int, run_status: str, run_id: str | None = None
) -> None:
    """Transition a node to its run's terminal status, idempotently and fenced.

    When `run_id` is given AND the node carries a run_id, the write goes through
    `graph.mark_terminal`, whose atomic `WHERE run_id = ? AND status = 'running'`
    is the real fence: a node re-claimed under a newer run_id (or already terminal)
    matches zero rows and is left for its current owner. This closes the TOCTOU a
    pre-check-then-unfenced-write left open — two concurrent reconcilers can both
    pass a Python-level run_id check, but only the matching atomic UPDATE lands.

    A node with no run_id (in-process TUI / legacy) has no fence to honour, so it
    falls back to the unconditional transition; only a RUNNING node is touched, so
    a node reset out of RUNNING is never force-marked. Shared by the sync (`mcp/run.py`)
    and detached worktree (`mcp/loop.py`) dispatch tools.
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
        row = graph.runs.get(run_id)
        attempt_id = row.get("attempt_id") if row is not None else None
        if target is NodeStatus.FAILED and isinstance(attempt_id, str) and row is not None:
            _reconcile_failed_attempt(graph, node, run_id, row)
            return
        _ = graph.mark_terminal(node_id, run_id, target)
    elif target is NodeStatus.DONE:
        graph.mark_done(node_id)
    else:
        graph.mark_failed(node_id)
