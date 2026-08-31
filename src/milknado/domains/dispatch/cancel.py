"""Run cancellation: cooperative async cancel, direct pid-run cancel, and finalize."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, cast

from milknado.domains.common import GitPort, MikadoNode, RunResult, UnlandedWorkError, pid_alive
from milknado.domains.dispatch._runstate import clear_cancel, now_iso, request_cancel, runs_dir
from milknado.domains.dispatch.ports import (
    ProcessTerminationPort,
    RunFinalizerPort,
    RunReaderPort,
)
from milknado.domains.dispatch.reconcile import fail_stale_running_runs, reconcile_node_status

if TYPE_CHECKING:
    from milknado.domains.graph import MikadoGraph

_logger = logging.getLogger(__name__)

# How long cancel waits for an async worker to observe the cancel sentinel and
# write its own terminal state before falling back to writing it directly. Must
# exceed the worker's worst-case finalize (poll interval + SIGTERM→SIGKILL grace
# in runner.py) so a live worker reliably wins the write.
_CANCEL_FINALIZE_TIMEOUT_SECS = 8.0
_CANCEL_FINALIZE_POLL_SECS = 0.25


def _finalize_cancelled(graph: RunFinalizerPort, run_id: str) -> dict[str, object]:
    """Write the cancelled terminal run row and return its refreshed dict.

    The direct-finalize path (pid run, or the fallback when the async worker never
    responds within the bound).
    """
    written = graph.finish_run(
        run_id,
        RunResult(
            status="failed",
            exit_code=-1,
            timed_out=False,
            ended_at=now_iso(),
            error="cancelled",
        ),
    )
    record = graph.get_run(run_id)
    if record is None or record.get("status") == "running":
        raise RuntimeError(
            f"run {run_id!r} cancellation finalization was not confirmed; "
            + "state and worktree preserved"
        )
    state: dict[str, object] = dict(record)
    if not written:
        state["terminal_persistence"] = "late-write-lost"
    return state


def _reconcile_cancel(
    graph: MikadoGraph,
    git: GitPort,
    node_id: int | None,
    run_id: str,
    status: str = "failed",
) -> Path | None:
    if node_id is None:
        return None
    preserved: Path | None = None
    node = graph.get_node(node_id)
    if node is None or node.run_id != run_id:
        # This run is not the node's current owner fence (e.g. a superseded
        # review-redispatch round). The run's own row is already finalized by
        # the caller; skip node-status/worktree reconciliation so a stale
        # round's cancel can't tear down a worktree or node another owner
        # still holds. Node-level lifecycle for adopted rounds belongs to the
        # Executor's own control loop, not this generic cancel path.
        return None
    if node.worktree_path:
        worktree = Path(node.worktree_path)
        if worktree.exists():
            try:
                git.remove_worktree(worktree)
            except Exception as exc:
                _logger.warning(
                    "cancel preserving worktree: run_id=%s worktree=%s: %s",
                    run_id,
                    worktree,
                    exc,
                    exc_info=not isinstance(exc, UnlandedWorkError),
                )
                preserved = worktree
    reconcile_node_status(graph, node_id, status, run_id=run_id)
    return preserved


def _await_cancel_finalize(graph: RunReaderPort, run_id: str) -> dict[str, object] | None:
    """Poll the run row for the async worker's terminal write, up to a bound.

    Returns the terminal run dict once the worker finalizes, or None if the bound
    elapses (the worker is wedged, dead, or never existed) so the caller can take
    over the terminal write itself.
    """
    deadline = time.monotonic() + _CANCEL_FINALIZE_TIMEOUT_SECS
    while time.monotonic() < deadline:
        time.sleep(_CANCEL_FINALIZE_POLL_SECS)
        record = graph.get_run(run_id)
        if record is not None and record.get("status") != "running":
            return dict(record)
    return None


def _reconcile_terminal_cancel(
    graph: MikadoGraph, git: GitPort, run_id: str, final: dict[str, object]
) -> dict[str, object]:
    preserved = _reconcile_cancel(
        graph,
        git,
        cast(int | None, final.get("node_id")),
        run_id,
        cast(str, final.get("status", "failed")),
    )
    final["worktree_preserved"] = str(preserved) if preserved is not None else None
    return final


def _adopt_pre_finalized_run(graph: MikadoGraph, git: GitPort, run_id: str) -> dict[str, object]:
    final = _await_cancel_finalize(graph, run_id)
    if final is None:
        record = graph.get_run(run_id)
        final = dict(record) if record is not None else None
    if final is None or final.get("status") == "running":
        raise RuntimeError(
            f"run {run_id!r} has no confirmed worker owner; state and worktree preserved"
        )
    return _reconcile_terminal_cancel(graph, git, run_id, final)


def _recover_dead_owner(
    graph: MikadoGraph, node: MikadoNode, node_id: int, run_id: str
) -> dict[str, object]:
    _ = fail_stale_running_runs(graph, node_id)
    record = graph.get_run(run_id)
    if record is None or record.get("status") == "running":
        raise RuntimeError(f"run {run_id!r} dead-owner recovery lost its terminal write")
    final: dict[str, object] = dict(record)
    reconcile_node_status(graph, node_id, cast(str, final["status"]), run_id=run_id)
    final["worktree_preserved"] = node.worktree_path
    return final


def _cancel_pid_run(
    graph: object,
    git: GitPort,
    process: ProcessTerminationPort,
    state: dict[str, object],
    run_id: str,
) -> dict[str, object]:
    pid = cast(int, state["pid"])
    if not process.terminate_group(pid, _CANCEL_FINALIZE_TIMEOUT_SECS):
        raise RuntimeError(
            f"run {run_id!r} did not exit after termination; state and worktree preserved"
        )
    typed_graph = cast("MikadoGraph", graph)
    final = _finalize_cancelled(typed_graph, run_id)
    preserved = _reconcile_cancel(typed_graph, git, cast(int | None, state.get("node_id")), run_id)
    final["worktree_preserved"] = str(preserved) if preserved is not None else None
    _logger.info(
        "run cancelled: run_id=%s node_id=%s substrate=process",
        run_id,
        state.get("node_id"),
    )
    return final


def _cancel_async_run(
    graph: RunReaderPort,
    git: GitPort,
    root: Path,
    state: dict[str, object],
    run_id: str,
) -> dict[str, object]:
    rdir = runs_dir(root)
    request_cancel(rdir, run_id)
    try:
        final = _await_cancel_finalize(graph, run_id)
        if final is None:
            latest = graph.get_run(run_id)
            if latest is None or latest.get("status") == "running":
                raise RuntimeError(
                    f"run {run_id!r} has not confirmed worker exit; state and worktree preserved"
                )
            final = dict(latest)
    finally:
        clear_cancel(rdir, run_id)
    final = _reconcile_terminal_cancel(cast("MikadoGraph", graph), git, run_id, final)
    _logger.info(
        "run cancelled: run_id=%s node_id=%s substrate=async",
        run_id,
        state.get("node_id"),
    )
    return final


def cancel_run(
    graph: MikadoGraph,
    git: GitPort,
    process: ProcessTerminationPort,
    project_root: Path,
    run_id: str,
) -> dict[str, object]:
    record = graph.get_run(run_id)
    if record is None:
        raise ValueError(f"run {run_id!r} not found")
    state: dict[str, object] = dict(record)
    if state.get("status") != "running":
        return state
    node_id = cast(int | None, state.get("node_id"))
    node = graph.get_node(node_id) if node_id is not None else None
    owner_pid = node.pid if node is not None and node.run_id == run_id else None
    run_pid = cast(int | None, state.get("pid"))
    if run_pid is not None:
        # A detached ralph run records the same process as its node owner and
        # can be terminated directly. Async runs retain the coordinator as the
        # node owner while their run row names the child worker, so they must
        # use the cooperative sentinel path instead of killing the coordinator's
        # process group.
        if owner_pid is not None and owner_pid != run_pid:
            return _cancel_async_run(graph, git, project_root, state, run_id)
        return _cancel_pid_run(graph, git, process, state, run_id)
    if owner_pid is None:
        # This never creates a cancel marker; a still-running row remains refused.
        return _adopt_pre_finalized_run(graph, git, run_id)
    if node is None or not isinstance(node_id, int):
        raise RuntimeError(
            f"run {run_id!r} has no confirmed node owner; state and worktree preserved"
        )
    if not pid_alive(owner_pid):
        return _recover_dead_owner(graph, node, node_id, run_id)
    raise RuntimeError(
        f"run {run_id!r} has no confirmed worker exit; state and worktree preserved"
    )
