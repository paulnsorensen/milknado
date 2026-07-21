"""Milknado MCP ralph-loop tools — detached, worktree-isolated ralph runs.

COORDINATOR-ONLY: never add these to WORKER_ALLOWED_TOOLS — a worker must not be
able to spawn sub-ralph-loops. Unlike milknado_run_inline* (a single-shot worker
piped a brief on stdin, isolated in its own worktree by default and merged back
on success — or run in the shared checkout via worktree=THIS_BRANCH), these
dispatch a node into its own git worktree + branch, iterate the ralph loop until
the node's quality gates pass, then rebase-merge the branch back — reusing the
same rebase-merge machinery run_inline's ISOLATE path now shares. The loop runs
in its own
detached process so it survives the MCP server restarting (hot-reload) or a
cloud env being reclaimed; run state and node status both live in SQLite (the
`runs` table and the `nodes` table), with log files on the filesystem.
"""

from __future__ import annotations

import logging
import os
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path

from milknado.adapters import GitAdapter, ProcessAdapter, TmuxAdapter, TmuxDispatchError
from milknado.domains.common import NodeKind, NodeStatus, RunResult
from milknado.domains.common.errors import UnlandedWorkError
from milknado.domains.dispatch import (
    RUN_ID_RE,
    ProcessPort,
    RunWindow,
    build_worker_env,
    ensure_tmux_ready,
    exit_code_path,
    fail_stale_running_runs,
    find_terminal_runs_for_node,
    latest_terminal_run,
    make_run_id,
    now_iso,
    reconcile_node_status,
    reconcile_run_window,
    runs_dir,
    tail,
)
from milknado.mcp._core import (
    build_run_dict,
    mcp,
    open_graph,
    resolve_project_root,
)

_logger = logging.getLogger(__name__)

_DEFAULT_RUNNER = (sys.executable, "-m", "milknado.mcp._ralph_node_runner")


def _resolve_runner_cmd(explicit: str | None) -> list[str]:
    if explicit and explicit.strip():
        return shlex.split(explicit)
    env = os.environ.get("MILKNADO_RALPH_RUNNER_CMD", "").strip()
    if env:
        return shlex.split(env)
    return list(_DEFAULT_RUNNER)


@dataclass(frozen=True)
class RalphStartRequest:
    node_id: int
    runner_cmd: str | None
    timeout_seconds: int
    use_tmux: bool
    root: Path


@dataclass(frozen=True)
class RalphClaim:
    run_id: str
    node_id: int
    target_branch: str
    base_oid: str
    stale_worktree: Path | None


def _claim_ralph(graph, git: GitAdapter, request: RalphStartRequest) -> RalphClaim:  # noqa: ANN001
    node = graph.get_node(request.node_id)
    if node is None:
        raise ValueError(f"node {request.node_id} not found")
    if node.kind != NodeKind.TASK:
        raise ValueError(
            f"node {request.node_id} has kind={node.kind.value}; only task nodes can be dispatched"
        )
    stale_worktree = Path(node.worktree_path) if node.worktree_path else None
    if node.status == NodeStatus.RUNNING:
        fail_stale_running_runs(graph, request.node_id)
        winner = latest_terminal_run(
            find_terminal_runs_for_node(graph, request.node_id, run_id=node.run_id)
        )
        if winner is not None:
            reconcile_node_status(
                graph,
                request.node_id,
                winner["status"],
                run_id=winner.get("run_id"),
            )
        graph.try_reclaim(request.node_id, now=now_iso())
    run_id = make_run_id(request.node_id)
    graph.claim_node_for_dispatch(request.node_id, run_id, now=now_iso())
    target_branch = git.current_branch()
    return RalphClaim(
        run_id=run_id,
        node_id=request.node_id,
        target_branch=target_branch,
        base_oid=git.resolve_ref(f"refs/heads/{target_branch}"),
        stale_worktree=stale_worktree,
    )


def _remove_reclaimed_worktree(git: GitAdapter, claim: RalphClaim) -> None:
    worktree = claim.stale_worktree
    if worktree is None or not worktree.exists():
        return
    try:
        git.remove_worktree(worktree)
    except UnlandedWorkError as exc:
        _logger.warning("Keeping orphan worktree (dispatch relocates): %s", exc)
    except Exception:
        _logger.exception(
            "Failed to remove reclaimed worktree: run_id=%s node_id=%d worktree=%s",
            claim.run_id,
            claim.node_id,
            worktree,
        )


def _runner_argv(request: RalphStartRequest, claim: RalphClaim) -> list[str]:
    return [
        *_resolve_runner_cmd(request.runner_cmd),
        "--node-id",
        str(request.node_id),
        "--project-root",
        str(request.root),
        "--run-id",
        claim.run_id,
        "--timeout",
        str(request.timeout_seconds),
        "--target-branch",
        claim.target_branch,
        "--base-oid",
        claim.base_oid,
    ]


def _spawn_ralph(
    process: ProcessPort,
    tmux: TmuxAdapter | None,
    request: RalphStartRequest,
    claim: RalphClaim,
    log_path: Path,
) -> int:
    argv = _runner_argv(request, claim)
    env = build_worker_env(
        {
            "MILKNADO_NODE_ID": str(request.node_id),
            "MILKNADO_RUN_ID": claim.run_id,
            "MILKNADO_PROJECT_ROOT": str(request.root),
        }
    )
    if tmux is not None:
        log_path.touch()
        return tmux.open_run_window(
            RunWindow(
                run_id=claim.run_id,
                argv=tuple(argv),
                cwd=request.root,
                log_path=log_path,
                exit_code_path=exit_code_path(runs_dir(request.root), claim.run_id),
                env=env,
            )
        )
    return process.spawn_detached(tuple(argv), request.root, log_path, env)


def _record_spawn_failure(graph, claim: RalphClaim, exc: Exception) -> None:  # noqa: ANN001
    try:
        graph.finish_run(
            claim.run_id,
            RunResult(
                status="failed",
                exit_code=-1,
                timed_out=False,
                ended_at=now_iso(),
                rebased=False,
                detail=f"spawn failed: {type(exc).__name__}: {exc}",
            ),
        )
    except Exception:
        _logger.exception(
            "spawn failure persistence failed: run_id=%s node_id=%d",
            claim.run_id,
            claim.node_id,
        )
    graph.mark_terminal(claim.node_id, claim.run_id, NodeStatus.FAILED)


@mcp.tool()
def milknado_run_loop_start(
    node_id: int,
    runner_cmd: str | None = None,
    timeout_seconds: int = 1800,
    use_tmux: bool = False,
    project_root: str = "",
) -> dict:
    """Start a task node in a detached worktree-backed Ralph loop.

    Returns immediately with a run ID for polling. Refuses concurrent dispatch.
    """
    root = resolve_project_root(project_root or None)
    request = RalphStartRequest(
        node_id=node_id,
        runner_cmd=runner_cmd,
        timeout_seconds=timeout_seconds,
        use_tmux=use_tmux,
        root=root,
    )
    tmux = TmuxAdapter(root) if use_tmux else None
    if tmux is not None:
        ensure_tmux_ready(tmux)
    graph, _cfg = open_graph(root)
    git = GitAdapter(root)
    try:
        claim = _claim_ralph(graph, git, request)
        _remove_reclaimed_worktree(git, claim)
        log_path = runs_dir(root) / f"{claim.run_id}.log"
        graph.start_run(
            claim.run_id,
            node_id,
            str(log_path),
            now_iso(),
            timeout_seconds,
        )
        try:
            pid = _spawn_ralph(ProcessAdapter(), tmux, request, claim, log_path)
        except (OSError, TmuxDispatchError) as exc:
            _record_spawn_failure(graph, claim, exc)
            raise
        graph.set_run_pid(claim.run_id, pid)
        graph.set_pid(node_id, claim.run_id, pid)
        _logger.info(
            "ralph dispatch started: run_id=%s node_id=%d pid=%d target_branch=%s base_oid=%s",
            claim.run_id,
            node_id,
            pid,
            claim.target_branch,
            claim.base_oid,
        )
        return build_run_dict(
            {
                "run_id": claim.run_id,
                "node_id": node_id,
                "status": "running",
                "pid": pid,
                "log_path": str(log_path),
            }
        )
    finally:
        graph.close()


@mcp.tool()
def milknado_run_loop_poll(run_id: str, project_root: str = "") -> dict:
    """Poll a ralph run started by milknado_run_loop_start.

    Returns the run state, rebase result, log tail, and summary fields without
    changing node status.
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
    if state.get("status") == "done":
        # Per-row window reconcile backstop: a completed run's window is
        # normally self-cleaned; kill a straggler (e.g. after a restart).
        reconcile_run_window(TmuxAdapter(root), state)
    # Derive the log path from the validated run_id rather than trusting the
    # stored field: no arbitrary-file read via a tampered log_path.
    state["summary"] = tail(rdir / f"{run_id}.log")
    return build_run_dict(state)
