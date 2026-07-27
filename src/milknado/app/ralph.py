"""Application-layer policy for detached, worktree-isolated ralph runs.

The MCP ``milknado_run_loop_start`` tool is a thin registration veneer over
``start_ralph_run`` here, which owns the claim/spawn policy and constructs the
git / process / tmux adapters. Entry modules therefore build no adapters and
hold no dispatch policy inline.
"""

from __future__ import annotations

import logging
import os
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path

from milknado.adapters import GitAdapter, ProcessAdapter, TmuxAdapter, TmuxDispatchError
from milknado.domains.common import NodeKind, NodeStatus, RunResult, UnlandedWorkError
from milknado.domains.dispatch import (
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
    runs_dir,
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


def _claim_ralph(graph, git: GitAdapter, request: RalphStartRequest) -> RalphClaim:
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


def _record_spawn_failure(graph, claim: RalphClaim, exc: Exception) -> None:
    run_written = False
    node_written = False
    persistence_error: Exception | None = None
    try:
        run_written = graph.finish_run(
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
    except Exception as error:
        persistence_error = error
        _logger.exception(
            "spawn failure run terminal write failed: run_id=%s node_id=%d",
            claim.run_id,
            claim.node_id,
        )
    try:
        node_written = graph.mark_terminal(claim.node_id, claim.run_id, NodeStatus.FAILED)
    except Exception as error:
        persistence_error = persistence_error or error
        _logger.exception(
            "spawn failure node terminal write failed: run_id=%s node_id=%d",
            claim.run_id,
            claim.node_id,
        )
    if run_written is False or node_written is False:
        detail = (
            f"spawn failure persistence failed: terminal writes incomplete: "
            f"run_written={run_written} node_written={node_written}"
        )
        _logger.error("%s run_id=%s node_id=%d", detail, claim.run_id, claim.node_id)
        raise RuntimeError(detail) from persistence_error


def start_ralph_run(graph, request: RalphStartRequest) -> dict:
    """Claim a task node and spawn its detached ralph loop; return the run state dict.

    Owns the adapter composition (git, process, tmux) and the claim/spawn policy
    so the MCP tool never constructs an adapter or holds this policy inline.
    """
    tmux = TmuxAdapter(request.root) if request.use_tmux else None
    if tmux is not None:
        ensure_tmux_ready(tmux)
    git = GitAdapter(request.root)
    claim = _claim_ralph(graph, git, request)
    _remove_reclaimed_worktree(git, claim)
    log_path = runs_dir(request.root) / f"{claim.run_id}.log"
    log_path.touch()
    graph.start_run(
        claim.run_id,
        request.node_id,
        str(log_path),
        now_iso(),
        request.timeout_seconds,
    )
    try:
        pid = _spawn_ralph(ProcessAdapter(), tmux, request, claim, log_path)
    except (OSError, TmuxDispatchError) as exc:
        _record_spawn_failure(graph, claim, exc)
        raise
    graph.set_run_pid(claim.run_id, pid)
    graph.set_pid(request.node_id, claim.run_id, pid)
    _logger.info(
        "ralph dispatch started: run_id=%s node_id=%d pid=%d target_branch=%s base_oid=%s",
        claim.run_id,
        request.node_id,
        pid,
        claim.target_branch,
        claim.base_oid,
    )
    return {
        "run_id": claim.run_id,
        "node_id": request.node_id,
        "status": "running",
        "pid": pid,
        "log_path": str(log_path),
    }
