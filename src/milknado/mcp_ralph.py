"""Milknado MCP ralph-loop tools — detached, worktree-isolated ralph runs.

COORDINATOR-ONLY: never add these to WORKER_ALLOWED_TOOLS — a worker must not be
able to spawn sub-ralph-loops. Unlike milknado_todo_run* (a single-shot worker
piped a brief on stdin, run in the shared working tree), these dispatch a node
into its own git worktree + branch, iterate the ralph loop until the node's
quality gates pass, then rebase-merge the branch back. The loop runs in its own
detached process so it survives the MCP server restarting (hot-reload) or a
cloud env being reclaimed; run state lives in `.milknado/runs/<run_id>.state.json`
and node status in SQLite.
"""

from __future__ import annotations

import logging
import os
import shlex
import subprocess
import sys
from contextlib import suppress

from milknado._mcp_core import mcp, open_graph, resolve_project_root
from milknado.domains.common import NodeStatus
from milknado.domains.dispatch import (
    fail_stale_running_runs,
    find_terminal_runs_for_node,
    latest_terminal_run,
    now_iso,
    read_state,
    reconcile_node_status,
    runs_dir,
    write_state,
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
def milknado_ralph_run_start(
    node_id: int,
    runner_cmd: str | None = None,
    timeout_seconds: int = 1800,
    project_root: str = "",
) -> dict:
    """Dispatch a node into its own git worktree and run a full ralph loop to
    completion in a detached process; returns immediately with a run_id.

    Atomically claims the node RUNNING in this parent process BEFORE spawning the
    detached runner — the SQLite conditional UPDATE is the cross-process
    mutual-exclusion point, so a second concurrent caller loses the claim and is
    refused. Creates a worktree + branch, iterates the ralph loop until the node's
    quality gates pass, then rebase-merges the branch back (the detached runner
    marks the node done/failed itself). The loop survives this server restarting.
    Poll with milknado_ralph_run_poll(run_id). A node left RUNNING by a dead runner
    is reclaimed by process-liveness check before the claim (no stale-timeout wait);
    orphaned runs that finished or vanished without a poll are reconciled first.
    """
    root = resolve_project_root(project_root or None)
    graph, _cfg = open_graph(root)
    try:
        node = graph.get_node(node_id)
        if node is None:
            raise ValueError(f"node {node_id} not found")
        now = now_iso()
        if node.status == NodeStatus.RUNNING:
            # First release a node locked by a run that finished, or whose detached
            # process vanished, without anyone polling (mirrors run_start).
            fail_stale_running_runs(root, node_id)
            winner = latest_terminal_run(find_terminal_runs_for_node(root, node_id))
            # Fence the reconcile on the terminal file's run_id: only the node's
            # current owner's run may drive its terminal transition, so a stale
            # terminal file from an older run cannot clobber a live run still
            # holding the node. A legacy node with no run_id has no fence to honour.
            if winner is not None and (node.run_id is None or winner.get("run_id") == node.run_id):
                reconcile_node_status(graph, node_id, winner["status"])
            # Then free a provably-dead owner by pid-liveness, so a crashed
            # runner does not lock the node for the full timeout (default 1800s).
            if graph.try_reclaim(node_id, now=now):
                _logger.info(
                    "milknado_ralph_run_start: reclaimed node=%d from a dead runner "
                    "(pid-liveness); re-dispatching without the stale-timeout wait",
                    node_id,
                )

        run_id = make_run_id(node_id)
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
        state_path = rdir / f"{run_id}.state.json"
        running_state = {
            "run_id": run_id,
            "node_id": node_id,
            "started_at": now,
            "log_path": str(log_path),
            "timeout_seconds": timeout_seconds,
            "status": "running",
            "rebased": None,
            "detail": None,
            "ended_at": None,
            "pid": None,
        }
        argv = [
            *_resolve_runner_cmd(runner_cmd),
            "--node-id",
            str(node_id),
            "--project-root",
            str(root),
            "--run-id",
            run_id,
            "--state-path",
            str(state_path),
            "--timeout",
            str(timeout_seconds),
        ]
        try:
            # Both the state-file write and the spawn are inside the guard: a
            # failure in either (fs permissions, partial .milknado setup, or an
            # unspawnable process) must release the claim, or the node is left
            # stranded RUNNING with no worker and no terminal state to reconcile.
            write_state(state_path, running_state)
            with log_path.open("wb") as log_fh:
                proc = subprocess.Popen(  # noqa: S603 — argv is built from validated parts
                    argv,
                    stdout=log_fh,
                    stderr=subprocess.STDOUT,
                    cwd=str(root),
                    start_new_session=True,
                    env=_build_worker_env({"MILKNADO_NODE_ID": str(node_id)}),
                )
        except OSError as exc:
            # Mark the node terminal via the fenced write that releases our claim,
            # then re-raise so the caller sees a clean failure. The terminal state
            # file is best-effort — it may not exist if write_state itself failed.
            terminal_state = {
                **running_state,
                "status": "failed",
                "rebased": False,
                "detail": f"spawn failed: {type(exc).__name__}: {exc}",
                "ended_at": now_iso(),
            }
            with suppress(OSError):
                write_state(state_path, terminal_state)
            graph.mark_terminal(node_id, run_id, NodeStatus.FAILED)
            raise
        # Record the detached pid on the node row (fenced on run_id) AND in the state
        # file so the next dispatch can liveness-check this run and reclaim it
        # immediately if it dies. Guard against a runner that already wrote its
        # terminal state, so the pid write never clobbers it back to "running".
        running = read_state(state_path)
        if running.get("status") == "running":
            running["pid"] = proc.pid
            write_state(state_path, running)
        graph.set_pid(node_id, run_id, proc.pid)
        _logger.info(
            "milknado_ralph_run_start: node=%d run_id=%s timeout=%ds pid=%d",
            node_id,
            run_id,
            timeout_seconds,
            proc.pid,
        )
        return {
            "run_id": run_id,
            "node_id": node_id,
            "status": "running",
            "pid": proc.pid,
            "log_path": str(log_path),
            "state_path": str(state_path),
        }
    finally:
        graph.close()


@mcp.tool()
def milknado_ralph_run_poll(run_id: str, project_root: str = "") -> dict:
    """Poll a ralph run started by milknado_ralph_run_start.

    Returns the run state (status running|done|failed, rebased, detail) plus a
    log tail. The detached runner reconciles node status itself via the executor,
    so unlike milknado_todo_run_poll this does not touch the graph.
    """
    root = resolve_project_root(project_root or None)
    if not RUN_ID_RE.match(run_id):
        raise ValueError(f"invalid run_id format: {run_id!r}")
    rdir = runs_dir(root)
    state_path = rdir / f"{run_id}.state.json"
    if not state_path.exists():
        raise ValueError(f"run {run_id!r} not found")
    state = read_state(state_path)
    # Derive the log path from the validated run_id rather than trusting the
    # stored field: no KeyError on a partial-write state, and no arbitrary-file
    # read via a tampered log_path.
    state["summary"] = tail(rdir / f"{run_id}.log")
    return state
