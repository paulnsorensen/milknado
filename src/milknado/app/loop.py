"""Application-layer policy for detached, worktree-isolated loop runs.

The MCP ``milknado_run_loop_start`` tool is a thin registration veneer over
``start_loop_run`` here, which owns the claim/spawn policy and constructs the
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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from milknado.domains.graph import MikadoGraph

from milknado.adapters import FlockSlotPool, GitAdapter, ProcessAdapter, TmuxAdapter
from milknado.app.worker_recovery import reconcile_loop_workers
from milknado.domains.common import (
    NodeKind,
    NodeStatus,
    RunFenceLostError,
    RunResult,
    SlotLease,
    UnlandedWorkError,
    pid_alive,
)
from milknado.domains.dispatch import (
    ProcessPort,
    ReapRequest,
    RunWindow,
    build_worker_env,
    claim_with_host_slot,
    ensure_tmux_ready,
    exit_code_path,
    fail_stale_running_runs,
    make_run_id,
    now_iso,
    reap_orphaned_workers,
    reconcile_node_status,
    reconcile_orphaned_runs,
    runs_dir,
)
from milknado.domains.graph import ConcurrencyLimitReached, NodeWorkers, HostCapacityFull

_logger = logging.getLogger(__name__)

_DEFAULT_RUNNER = (sys.executable, "-m", "milknado.mcp._loop_node_runner")


def _resolve_runner_cmd(explicit: str | None) -> list[str]:
    if explicit and explicit.strip():
        return shlex.split(explicit)
    env = os.environ.get("MILKNADO_LOOP_RUNNER_CMD", "").strip()
    if env:
        return shlex.split(env)
    return list(_DEFAULT_RUNNER)


@dataclass(frozen=True)
class LoopStartRequest:
    node_id: int
    runner_cmd: str | None
    timeout_seconds: int
    use_tmux: bool
    root: Path
    host_worker_limit: int | None = None


@dataclass(frozen=True)
class LoopClaim:
    run_id: str
    node_id: int
    target_branch: str
    base_oid: str
    stale_worktree: Path | None
    lease: SlotLease | None = None


def _claim_loop(graph: MikadoGraph, git: GitAdapter, request: LoopStartRequest) -> LoopClaim:
    node = graph.get_node(request.node_id)
    if node is None:
        raise ValueError(f"node {request.node_id} not found")
    if node.kind != NodeKind.TASK:
        raise ValueError(
            f"node {request.node_id} has kind={node.kind.value}; only task nodes can be dispatched"
        )
    stale_worktree = Path(node.worktree_path) if node.worktree_path else None
    if node.status == NodeStatus.RUNNING:
        process = ProcessAdapter()
        if (
            node.pid is not None
            and not pid_alive(node.pid)
            and not reap_orphaned_workers(
                graph, process, ReapRequest(NodeWorkers(request.node_id))
            )
        ):
            raise RuntimeError(
                f"node {request.node_id} worker recovery unresolved; claim and worktree preserved"
            )
        _ = fail_stale_running_runs(graph, request.node_id, process)
        if node.run_id is not None:
            winner = graph.runs.latest_terminal(request.node_id, node.run_id)
            if winner is not None:
                reconcile_node_status(
                    graph,
                    request.node_id,
                    winner["status"],
                    run_id=winner.get("run_id"),
                )
        else:
            orphan = graph.runs.latest_unowned_terminal(request.node_id)
            if orphan is not None:
                reconcile_node_status(graph, request.node_id, orphan["status"])
        _ = graph.try_reclaim(request.node_id, now=now_iso())
    target_branch = git.current_branch()
    base_oid = git.resolve_ref(f"refs/heads/{target_branch}")
    run_id = make_run_id(request.node_id)
    pool = FlockSlotPool(request.host_worker_limit) if request.host_worker_limit else None
    lease = claim_with_host_slot(graph, pool, (request.node_id, run_id), request.root)
    return LoopClaim(
        run_id=run_id,
        node_id=request.node_id,
        target_branch=target_branch,
        base_oid=base_oid,
        stale_worktree=stale_worktree,
        lease=lease,
    )


def _remove_reclaimed_worktree(git: GitAdapter, claim: LoopClaim) -> None:
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


def _runner_argv(request: LoopStartRequest, claim: LoopClaim) -> list[str]:
    return [
        *_resolve_runner_cmd(request.runner_cmd),
        "--node-id",
        str(request.node_id),
        "--project-root",
        str(request.root),
        "--run-id",
        claim.run_id,
        "--target-branch",
        claim.target_branch,
        "--base-oid",
        claim.base_oid,
    ]


def _spawn_loop(
    process: ProcessPort,
    tmux: TmuxAdapter | None,
    request: LoopStartRequest,
    claim: LoopClaim,
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


def _record_start_failure(
    graph: MikadoGraph, claim: LoopClaim, exc: Exception, *, run_started: bool
) -> None:
    node_written = False
    persistence_error: Exception | None = None
    if run_started:
        try:
            graph.runs.finish(
                claim.run_id,
                RunResult(
                    status="failed",
                    exit_code=-1,
                    timed_out=False,
                    ended_at=now_iso(),
                    rebased=False,
                    detail=f"start failed: {type(exc).__name__}: {exc}",
                ),
            )
        except RunFenceLostError:
            _logger.info("start failure run already finalized: %s", claim.run_id)
        except Exception as error:
            persistence_error = error
            _logger.exception("start failure run terminal write failed: run_id=%s", claim.run_id)
    try:
        node_written = graph.mark_terminal(claim.node_id, claim.run_id, NodeStatus.FAILED)
    except Exception as error:
        persistence_error = persistence_error or error
        _logger.exception("start failure node terminal write failed: node_id=%d", claim.node_id)
    if persistence_error is not None or node_written is False:
        detail = (
            f"start failure persistence failed: terminal writes incomplete: "
            f"node_written={node_written}"
        )
        _logger.error("%s run_id=%s node_id=%d", detail, claim.run_id, claim.node_id)
        raise RuntimeError(detail) from persistence_error


def start_loop_run(graph: MikadoGraph, request: LoopStartRequest) -> dict[str, object]:
    """Claim a task node and spawn its detached loop; return the run state dict.

    Returns a deferred result instead when the graph is at its concurrency limit.
    Owns the adapter composition (git, process, tmux) and the claim/spawn policy
    so the MCP tool never constructs an adapter or holds this policy inline.
    """
    graph.register_controller_master()
    reconcile_loop_workers(graph)
    _ = reconcile_orphaned_runs(graph, ProcessAdapter())
    tmux = TmuxAdapter(request.root) if request.use_tmux else None
    if tmux is not None:
        ensure_tmux_ready(tmux)
    git = GitAdapter(request.root)
    try:
        claim = _claim_loop(graph, git, request)
    except ConcurrencyLimitReached as exc:
        host_full = isinstance(exc, HostCapacityFull)
        _logger.info(
            "loop dispatch deferred: node_id=%d running=%d limit=%d",
            request.node_id,
            exc.running,
            exc.limit,
        )
        return {
            "node_id": request.node_id,
            "status": "deferred",
            "running": exc.running,
            "limit": exc.limit,
            "detail": (
                f"host worker pool full ({exc.running}/{exc.limit}), possibly held by "
                + "other projects; wait for a worker to finish, then start this node again"
                if host_full
                else f"concurrency limit reached: {exc.running} of {exc.limit} tasks are "
                + "running; wait for a task to finish, then start this node again"
            ),
        }
    run_started = False
    try:
        _remove_reclaimed_worktree(git, claim)
        log_path = runs_dir(request.root) / f"{claim.run_id}.log"
        log_path.touch()
        graph.runs.start(
            claim.run_id,
            request.node_id,
            str(log_path),
            now_iso(),
            request.timeout_seconds,
        )
        run_started = True
        pid = _spawn_loop(ProcessAdapter(), tmux, request, claim, log_path)
    except Exception as exc:
        _record_start_failure(graph, claim, exc, run_started=run_started)
        raise
    finally:
        if claim.lease is not None:
            claim.lease.release()
    graph.runs.set_pid(claim.run_id, pid)
    graph.set_pid(request.node_id, claim.run_id, pid)
    _logger.info(
        "loop dispatch started: run_id=%s node_id=%d pid=%d target_branch=%s base_oid=%s",
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
