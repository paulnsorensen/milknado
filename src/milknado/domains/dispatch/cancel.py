"""Run cancellation: cooperative async cancel, direct pid-run cancel, and finalize."""

from __future__ import annotations

import logging
import os
import signal
import time
from pathlib import Path

from milknado.adapters import GitAdapter
from milknado.domains.dispatch._runstate import clear_cancel, now_iso, request_cancel, runs_dir
from milknado.domains.dispatch.reconcile import reconcile_node_status

_logger = logging.getLogger(__name__)

# How long cancel waits for an async worker to observe the cancel sentinel and
# write its own terminal state before falling back to writing it directly. Must
# exceed the worker's worst-case finalize (poll interval + SIGTERM→SIGKILL grace
# in runner.py) so a live worker reliably wins the write.
_CANCEL_FINALIZE_TIMEOUT_SECS = 8.0
_CANCEL_FINALIZE_POLL_SECS = 0.25


def _open_graph_for_cancel(root: Path):  # noqa: ANN202
    """Open a graph connection for cancel's reconcile step.

    Imported lazily to avoid a domains -> _mcp_core import cycle.
    """
    from milknado._mcp_core import open_graph

    graph, _cfg = open_graph(root)
    return graph


def _finalize_cancelled(graph, run_id: str) -> dict:  # noqa: ANN001
    """Write the cancelled terminal run row and return its refreshed dict.

    The direct-finalize path (pid run, or the fallback when the async worker never
    responds within the bound).
    """
    graph.finish_run(
        run_id,
        status="failed",
        exit_code=-1,
        timed_out=False,
        ended_at=now_iso(),
        error="cancelled",
    )
    return graph.get_run(run_id)


def _reconcile_cancel(root: Path, node_id, run_id: str, status: str = "failed") -> None:  # noqa: ANN001
    """Prune the run's worktree and reconcile its node to `status`, fenced on run_id.

    Routes through the run_id-fenced `reconcile_node_status` (graph.mark_terminal)
    so a node re-claimed under a newer run cannot be clobbered by a stale cancel —
    the merged worker-dispatch fencing invariant. A node with no run_id falls back
    to the unconditional RUNNING→terminal transition. `status` is the run's actual
    terminal status: a worker that finished `done` inside the cancel window must
    reconcile its node `done`, not be force-failed.
    """
    if node_id is None:
        return
    graph = _open_graph_for_cancel(root)
    try:
        node = graph.get_node(node_id)
        if node is not None and node.worktree_path:
            wt = Path(node.worktree_path)
            if wt.exists():
                try:
                    GitAdapter(root).remove_worktree(wt)
                except Exception as exc:
                    _logger.warning("Failed to remove worktree %s on cancel: %s", wt, exc)
        reconcile_node_status(graph, node_id, status, run_id=run_id)
    finally:
        graph.close()
    try:
        GitAdapter(root).prune_worktrees()
    except Exception as exc:
        _logger.warning("git worktree prune failed on cancel: %s", exc)


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


def _cancel_pid_run(graph, root: Path, state: dict, run_id: str) -> dict:  # noqa: ANN001
    """Cancel a detached-ralph run: its own process group → killpg is safe, and the
    run owns no in-process worker, so finalize the terminal state here directly."""
    pid = state["pid"]
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except ProcessLookupError:
        pass  # process already gone — fall through to finalize the terminal state
    # Any other OSError (e.g. EPERM) means the process may still be alive: let it
    # propagate rather than reporting a run cancelled that we never signalled.
    final = _finalize_cancelled(graph, run_id)
    _reconcile_cancel(root, state.get("node_id"), run_id)
    _logger.info("cancel(pid): run_id=%s node_id=%s", run_id, state.get("node_id"))
    return final


def _cancel_async_run(graph, root: Path, state: dict, run_id: str) -> dict:  # noqa: ANN001
    """Cancel an async-headless run (or a crashed run stuck "running"). Request
    cooperative cancellation and let the worker own the terminal write; only take
    over if the worker never finalizes within the bound."""
    rdir = runs_dir(root)
    request_cancel(rdir, run_id)
    final = _await_cancel_finalize(graph, run_id)
    if final is None:
        # Worker never finalized within the bound: it is wedged, dead, or was never
        # alive (crashed sync run). Take over the terminal write so the node is not
        # stranded RUNNING, and clear the sentinel we wrote.
        latest = graph.get_run(run_id)
        if latest is not None and latest.get("status") != "running":
            final = latest  # worker finalized in the last poll gap — don't clobber
        else:
            final = _finalize_cancelled(graph, run_id)
        clear_cancel(rdir, run_id)

    # Reconcile to the run's ACTUAL terminal status: a worker that finished `done`
    # inside the finalize window must not have its node force-failed.
    _reconcile_cancel(root, final.get("node_id"), run_id, final.get("status", "failed"))
    _logger.info("cancel(async): run_id=%s node_id=%s", run_id, state.get("node_id"))
    return final


def cancel_run(graph, project_root: Path, run_id: str) -> dict:  # noqa: ANN001
    """Cancel a run, reconcile node status, and prune any worktree.

    A detached-ralph run (has a recorded process-group pid) is signalled with
    SIGTERM and finalized immediately. An async-headless run shares the MCP
    server's process group, so it cannot be signalled — instead a cancel sentinel
    is written and the worker cooperatively terminates its own subprocess and
    writes the terminal state, which closes the state-clobber race. Cancel waits a
    bounded window for that finalize, then takes over the terminal write only if
    the worker never responds (wedged or crashed). No-ops cleanly if the run is
    already terminal. Returns the final run state.
    """
    state = graph.get_run(run_id)
    if state is None:
        raise ValueError(f"run {run_id!r} not found")
    if state.get("status") != "running":
        return state
    if state.get("pid") is not None:
        return _cancel_pid_run(graph, project_root, state, run_id)
    return _cancel_async_run(graph, project_root, state, run_id)
