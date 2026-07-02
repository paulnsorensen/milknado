"""Milknado MCP ralph-loop tools — detached, worktree-isolated ralph runs.

COORDINATOR-ONLY: never add these to WORKER_ALLOWED_TOOLS — a worker must not be
able to spawn sub-ralph-loops. Unlike milknado_run_once* (a single-shot worker
piped a brief on stdin, run in the shared working tree), these dispatch a node
into its own git worktree + branch, iterate the ralph loop until the node's
quality gates pass, then rebase-merge the branch back. The loop runs in its own
detached process so it survives the MCP server restarting (hot-reload) or a
cloud env being reclaimed; run state and node status both live in SQLite (the
`runs` table and the `nodes` table), with log files on the filesystem.
"""

from __future__ import annotations

import logging
import os
import shlex
import subprocess
import sys
from contextlib import suppress
from pathlib import Path

from milknado._mcp_core import (
    _check_ancestor_goal_not_claimed,
    _claim_ancestor_goal_for_dispatch,
    mcp,
    open_graph,
    resolve_project_root,
)
from milknado.adapters import GitAdapter
from milknado.domains.common import NodeKind, NodeStatus
from milknado.domains.common.errors import UnlandedWorkError
from milknado.domains.dispatch import (
    fail_stale_running_runs,
    find_terminal_runs_for_node,
    latest_terminal_run,
    now_iso,
    reconcile_node_status,
    runs_dir,
)
from milknado.domains.dispatch._runstate import (
    RUN_ID_RE,
    make_run_id,
    tail,
)
from milknado.domains.dispatch.runner import _build_worker_env

_logger = logging.getLogger(__name__)

_DEFAULT_RUNNER = (sys.executable, "-m", "milknado._ralph_node_runner")


def _resolve_runner_cmd(explicit: str | None) -> list[str]:
    if explicit and explicit.strip():
        return shlex.split(explicit)
    env = os.environ.get("MILKNADO_RALPH_RUNNER_CMD", "").strip()
    if env:
        return shlex.split(env)
    return list(_DEFAULT_RUNNER)


@mcp.tool()
def milknado_run_loop_start(
    node_id: int,
    runner_cmd: str | None = None,
    timeout_seconds: int = 1800,
    project_root: str = "",
) -> dict:
    """Claim node_id and spawn a detached ralph loop; returns immediately with a run_id.

    A second concurrent caller is refused (atomic claim). Poll with
    milknado_run_loop_poll(run_id).
    """
    root = resolve_project_root(project_root or None)
    graph, _cfg = open_graph(root)
    try:
        node = graph.get_node(node_id)
        if node is None:
            raise ValueError(f"node {node_id} not found")
        if node.kind != NodeKind.TASK:
            raise ValueError(
                f"node {node_id} has kind={node.kind.value}; only task nodes can be dispatched"
            )
        run_id = make_run_id(node_id)
        _check_ancestor_goal_not_claimed(graph, node_id)
        _claim_ancestor_goal_for_dispatch(graph, node_id, run_id)
        now = now_iso()
        if node.status == NodeStatus.RUNNING:
            # Capture the worktree path BEFORE reconcile clears it from the node row,
            # so we can prune an orphaned worktree left by a killed runner before
            # spawning a new one — git-worktree-add fails if the path still exists.
            orphan_wt = Path(node.worktree_path) if node.worktree_path else None
            # First release a node locked by a run that finished, or whose detached
            # process vanished, without anyone polling (mirrors run_start).
            fail_stale_running_runs(graph, node_id)
            # Filter terminal files to this node's fence BEFORE picking the latest,
            # so a stale run with a later ended_at cannot mask the current owner's
            # file. reconcile_node_status routes the write through the atomic
            # run_id + status fence (graph.mark_terminal), closing the TOCTOU a
            # pre-check-then-unfenced-write left open: a node re-claimed under a
            # newer run cannot be clobbered. A legacy node with no run_id has no
            # fence to honour and falls back to the unconditional reconcile.
            winner = latest_terminal_run(
                find_terminal_runs_for_node(graph, node_id, run_id=node.run_id)
            )
            if winner is not None:
                reconcile_node_status(
                    graph, node_id, winner["status"], run_id=winner.get("run_id")
                )
            # Then free a provably-dead owner by pid-liveness, so a crashed
            # runner does not lock the node for the full timeout (default 1800s).
            if graph.try_reclaim(node_id, now=now):
                _logger.info(
                    "milknado_run_loop_start: reclaimed node=%d from a dead runner "
                    "(pid-liveness); re-dispatching without the stale-timeout wait",
                    node_id,
                )
            # Prune the orphaned worktree so git-worktree-add can reuse the path.
            # Fail-closed: a dirty/unlanded orphan refuses removal and is kept —
            # the dispatch relocates to a suffixed path instead of blocking the
            # run loop (see executor._relocate_occupied).
            if orphan_wt is not None and orphan_wt.exists():
                try:
                    GitAdapter(root).remove_worktree(orphan_wt)
                except UnlandedWorkError as exc:
                    _logger.warning("Keeping orphan worktree (dispatch relocates): %s", exc)
                except Exception as exc:
                    _logger.warning("Failed to remove orphan worktree %s: %s", orphan_wt, exc)

        if not graph.claim_node(node_id, run_id, now=now):
            current = graph.get_node(node_id)
            status = current.status.value if current is not None else "gone"
            raise ValueError(
                f"node {node_id} is already {status}; set status back to pending to retry"
            )
        # The node is RUNNING under our run_id BEFORE we spawn — a second concurrent
        # caller's claim above already lost. From here a failure must release the
        # claim (mark_terminal failed) so the node is not stranded RUNNING.
        rdir = runs_dir(root)
        log_path = rdir / f"{run_id}.log"
        argv = [
            *_resolve_runner_cmd(runner_cmd),
            "--node-id",
            str(node_id),
            "--project-root",
            str(root),
            "--run-id",
            run_id,
            "--timeout",
            str(timeout_seconds),
        ]
        try:
            # Both the run-row insert and the spawn are inside the guard: a failure
            # in either (fs permissions, partial .milknado setup, or an unspawnable
            # process) must release the claim, or the node is left stranded RUNNING
            # with no worker and no terminal run to reconcile.
            graph.start_run(run_id, node_id, str(log_path), now, timeout_seconds)
            with log_path.open("wb") as log_fh:
                proc = subprocess.Popen(  # noqa: S603 — COORDINATOR-ONLY module (see docstring); argv is coordinator-assembled, not external input
                    argv,
                    stdout=log_fh,
                    stderr=subprocess.STDOUT,
                    cwd=str(root),
                    start_new_session=True,
                    env=_build_worker_env(
                        {"MILKNADO_NODE_ID": str(node_id), "MILKNADO_RUN_ID": run_id}
                    ),
                )
        except OSError as exc:
            # Mark the node terminal via the fenced write that releases our claim,
            # then re-raise so the caller sees a clean failure. The terminal run
            # write is best-effort — the run row exists only if start_run landed.
            with suppress(Exception):
                graph.finish_run(
                    run_id,
                    status="failed",
                    exit_code=-1,
                    timed_out=False,
                    ended_at=now_iso(),
                    rebased=False,
                    detail=f"spawn failed: {type(exc).__name__}: {exc}",
                )
            graph.mark_terminal(node_id, run_id, NodeStatus.FAILED)
            raise
        # Record the detached pid on the node row (fenced on run_id) AND on the run
        # row so the next dispatch can liveness-check this run and reclaim it
        # immediately if it dies. set_run_pid is gated on status='running', so the
        # pid write never clobbers a runner that already wrote its terminal state.
        graph.set_run_pid(run_id, proc.pid)
        graph.set_pid(node_id, run_id, proc.pid)
        _logger.info(
            "milknado_run_loop_start: node=%d run_id=%s timeout=%ds pid=%d",
            node_id,
            run_id,
            timeout_seconds,
            proc.pid,
        )
        return {
            "run_id": run_id,
            "node_id": node_id,
            "status": "running",
            "exit_code": None,
            "timed_out": None,
            "rebased": None,
            "pid": proc.pid,
            "log_path": str(log_path),
            "summary": None,
        }
    finally:
        graph.close()


@mcp.tool()
def milknado_run_loop_poll(run_id: str, project_root: str = "") -> dict:
    """Poll a ralph run started by milknado_run_loop_start.

    Returns run state (status running|done|failed, rebased, detail) plus a log tail.
    """
    root = resolve_project_root(project_root or None)
    if not RUN_ID_RE.match(run_id):
        raise ValueError(f"invalid run_id format: {run_id!r}")
    rdir = runs_dir(root)
    graph, _cfg = open_graph(root)
    try:
        state = graph.get_run(run_id)
    finally:
        graph.close()
    if state is None:
        raise ValueError(f"run {run_id!r} not found")
    # Derive the log path from the validated run_id rather than trusting the
    # stored field: no arbitrary-file read via a tampered log_path.
    state["summary"] = tail(rdir / f"{run_id}.log")
    state.setdefault("exit_code", None)
    state.setdefault("timed_out", None)
    return state
