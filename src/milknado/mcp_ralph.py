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

import os
import shlex
import subprocess
import sys

from milknado._mcp_core import mcp, open_graph, resolve_project_root
from milknado.domains.common import NodeStatus
from milknado.domains.dispatch import fail_stale_running_runs, find_terminal_runs_for_node
from milknado.domains.dispatch._runstate import (
    RUN_ID_RE,
    make_run_id,
    now_iso,
    read_state,
    runs_dir,
    tail,
    write_state,
)
from milknado.mcp_run import _reconcile_node_status

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

    Creates a worktree + branch, iterates the ralph loop until the node's quality
    gates pass, then rebase-merges the branch back (the detached runner marks the
    node done/failed itself). The loop survives this server restarting. Poll with
    milknado_ralph_run_poll(run_id). Refuses if the node is already running;
    orphaned runs that finished or vanished without a poll are reconciled first.
    """
    root = resolve_project_root(project_root or None)
    graph, _cfg = open_graph(root)
    try:
        node = graph.get_node(node_id)
        if node is None:
            raise ValueError(f"node {node_id} not found")
        if node.status == NodeStatus.RUNNING:
            # Release a node locked by a run that finished, or whose detached
            # process vanished, without anyone polling (mirrors run_start).
            fail_stale_running_runs(root, node_id)
            for state in find_terminal_runs_for_node(root, node_id):
                _reconcile_node_status(graph, node_id, state["status"])
            node = graph.get_node(node_id)
            assert node is not None
            if node.status == NodeStatus.RUNNING:
                raise ValueError(
                    f"node {node_id} is already running; set status back to pending to retry"
                )

        run_id = make_run_id(node_id)
        rdir = runs_dir(root)
        log_path = rdir / f"{run_id}.log"
        state_path = rdir / f"{run_id}.state.json"
        write_state(
            state_path,
            {
                "run_id": run_id,
                "node_id": node_id,
                "started_at": now_iso(),
                "log_path": str(log_path),
                "timeout_seconds": timeout_seconds,
                "status": "running",
                "rebased": None,
                "detail": None,
                "ended_at": None,
            },
        )
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
            with log_path.open("wb") as log_fh:
                subprocess.Popen(  # noqa: S603 — argv is built from validated parts
                    argv,
                    stdout=log_fh,
                    stderr=subprocess.STDOUT,
                    cwd=str(root),
                    start_new_session=True,
                )
        except OSError as exc:
            # A bad runner_cmd or an unspawnable process would otherwise leave
            # the state file stuck at "running" forever. Mark it terminal and
            # re-raise so the caller sees a clean failure.
            failed = read_state(state_path)
            failed.update(
                status="failed",
                rebased=False,
                detail=f"spawn failed: {type(exc).__name__}: {exc}",
                ended_at=now_iso(),
            )
            write_state(state_path, failed)
            raise
        return {
            "run_id": run_id,
            "node_id": node_id,
            "status": "running",
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
