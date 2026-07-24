"""Run cancellation: cooperative async cancel, direct pid-run cancel, and finalize."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from milknado.domains.common import GitPort, RunResult, UnlandedWorkError
from milknado.domains.dispatch._runstate import clear_cancel, now_iso, request_cancel, runs_dir
from milknado.domains.dispatch.ports import ProcessTerminationPort
from milknado.domains.dispatch.reconcile import reconcile_node_status

_logger = logging.getLogger(__name__)

# How long cancel waits for an async worker to observe the cancel sentinel and
# write its own terminal state before falling back to writing it directly. Must
# exceed the worker's worst-case finalize (poll interval + SIGTERM→SIGKILL grace
# in runner.py) so a live worker reliably wins the write.
_CANCEL_FINALIZE_TIMEOUT_SECS = 8.0
_CANCEL_FINALIZE_POLL_SECS = 0.25


def _finalize_cancelled(graph, run_id: str) -> dict:  # noqa: ANN001
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
    state = graph.get_run(run_id)
    if state is not None and not written:
        state["terminal_persistence"] = "late-write-lost"
    return state


def _reconcile_cancel(
    graph,  # noqa: ANN001
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


def _await_cancel_finalize(graph, run_id: str) -> dict | None:  # noqa: ANN001
    """Poll the run row for the async worker's terminal write, up to a bound.

    Returns the terminal run dict once the worker finalizes, or None if the bound
    elapses (the worker is wedged, dead, or never existed) so the caller can take
    over the terminal write itself.
    """
    deadline = time.monotonic() + _CANCEL_FINALIZE_TIMEOUT_SECS
    while time.monotonic() < deadline:
        time.sleep(_CANCEL_FINALIZE_POLL_SECS)
        state = graph.get_run(run_id)
        if state is not None and state.get("status") != "running":
            return state
    return None


def _cancel_pid_run(
    graph,  # noqa: ANN001
    git: GitPort,
    process: ProcessTerminationPort,
    state: dict,
    run_id: str,
) -> dict:
    pid = state["pid"]
    if not process.terminate_group(pid, _CANCEL_FINALIZE_TIMEOUT_SECS):
        raise RuntimeError(
            f"run {run_id!r} did not exit after termination; state and worktree preserved"
        )
    final = _finalize_cancelled(graph, run_id)
    preserved = _reconcile_cancel(graph, git, state.get("node_id"), run_id)
    final["worktree_preserved"] = str(preserved) if preserved is not None else None
    _logger.info(
        "run cancelled: run_id=%s node_id=%s substrate=process",
        run_id,
        state.get("node_id"),
    )
    return final


def _cancel_async_run(
    graph,  # noqa: ANN001
    git: GitPort,
    root: Path,
    state: dict,
    run_id: str,
) -> dict:
    rdir = runs_dir(root)
    request_cancel(rdir, run_id)
    final = _await_cancel_finalize(graph, run_id)
    if final is None:
        latest = graph.get_run(run_id)
        if latest is None or latest.get("status") == "running":
            raise RuntimeError(
                f"run {run_id!r} has not confirmed worker exit; state and worktree preserved"
            )
        final = latest
    clear_cancel(rdir, run_id)
    preserved = _reconcile_cancel(
        graph, git, final.get("node_id"), run_id, final.get("status", "failed")
    )
    final["worktree_preserved"] = str(preserved) if preserved is not None else None
    _logger.info(
        "run cancelled: run_id=%s node_id=%s substrate=async",
        run_id,
        state.get("node_id"),
    )
    return final


def cancel_run(
    graph,  # noqa: ANN001
    git: GitPort,
    process: ProcessTerminationPort,
    project_root: Path,
    run_id: str,
) -> dict:
    state = graph.get_run(run_id)
    if state is None:
        raise ValueError(f"run {run_id!r} not found")
    if state.get("status") != "running":
        return state
    if state.get("pid") is not None:
        return _cancel_pid_run(graph, git, process, state, run_id)
    return _cancel_async_run(graph, git, project_root, state, run_id)
