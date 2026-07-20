"""Application-layer ralph dispatch: claim, spawn, and subprocess node runner."""

from __future__ import annotations

import logging
import os
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from milknado.adapters import ProcessAdapter, TmuxAdapter

if TYPE_CHECKING:
    from milknado.adapters import GitAdapter

from milknado.domains.common import NodeKind, NodeStatus, RunResult
from milknado.domains.common.errors import UnlandedWorkError
from milknado.domains.dispatch import (
    RUN_ID_RE,  # noqa: F401 — re-exported for tests
    ProcessPort,
    RunWindow,
    build_worker_env,
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


def _resolve_runner_cmd(explicit: str | None) -> list[str]:
    if explicit and explicit.strip():
        return shlex.split(explicit)
    env = os.environ.get("MILKNADO_RALPH_RUNNER_CMD", "").strip()
    if env:
        return shlex.split(env)
    return list(_DEFAULT_RUNNER)


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
                graph, request.node_id, winner["status"], run_id=winner.get("run_id")
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
    tmux,  # noqa: ANN001
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


def start_ralph_run(graph, request: RalphStartRequest) -> dict:  # noqa: ANN001
    """Claim a node, remove stale worktree, start the run row, and spawn the subprocess."""
    from milknado.adapters import GitAdapter
    from milknado.adapters.tmux import TmuxDispatchError
    from milknado.domains.dispatch import ensure_tmux_ready

    git = GitAdapter(request.root)
    tmux = TmuxAdapter(request.root) if request.use_tmux else None
    if tmux is not None:
        ensure_tmux_ready(tmux)

    claim = _claim_ralph(graph, git, request)
    _remove_reclaimed_worktree(git, claim)
    log_path = runs_dir(request.root) / f"{claim.run_id}.log"
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


@dataclass(frozen=True)
class RunNodeRequest:
    root: Path
    node_id: int
    run_id: str
    timeout: float
    target_branch: str
    base_oid: str


def run_node_subprocess(request: RunNodeRequest) -> int:
    """Run a node to completion; called by the headless node runner process."""
    from milknado.adapters import CrgAdapter, GitAdapter, LoopAdapter
    from milknado.app.project import open_graph
    from milknado.domains.common import resolve_flavor_profile
    from milknado.domains.execution import (
        NO_GATES_CONFIGURED_MESSAGE,
        ExecutionConfig,
        Executor,
        run_node_to_completion,
    )

    _logger.info(
        "ralph runner started: run_id=%s node_id=%d target_branch=%s base_oid=%s",
        request.run_id,
        request.node_id,
        request.target_branch,
        request.base_oid,
    )
    graph, cfg = open_graph(request.root)
    try:
        node = graph.get_node(request.node_id)
        profile = resolve_flavor_profile(cfg, node.flavor if node is not None else None)
        if profile.quality_gates is None:
            _logger.error(
                "ralph preflight failed: run_id=%s node_id=%d error=%s",
                request.run_id,
                request.node_id,
                NO_GATES_CONFIGURED_MESSAGE,
            )
            graph.finish_run(
                request.run_id,
                RunResult(
                    status="failed",
                    exit_code=1,
                    timed_out=False,
                    ended_at=now_iso(),
                    rebased=False,
                    detail=NO_GATES_CONFIGURED_MESSAGE,
                ),
            )
            return 1
        git = GitAdapter(request.root)
        ralph = LoopAdapter()
        crg = CrgAdapter(request.root)
        executor = Executor(graph=graph, git=git, ralph=ralph, crg=crg)
        exec_config = ExecutionConfig(
            execution_agent=profile.execution_agent,
            quality_gates=profile.quality_gates,
            worktree_pattern=cfg.worktree_pattern,
            project_root=request.root,
            commit_footer=cfg.commit_footer,
        )
        outcome = run_node_to_completion(
            executor,
            ralph,
            request.node_id,
            exec_config,
            request.target_branch,
            request.timeout,
            base_oid=request.base_oid,
        )
        graph.finish_run(
            request.run_id,
            RunResult(
                status="done" if outcome.success else "failed",
                exit_code=0 if outcome.success else 1,
                timed_out=False,
                ended_at=now_iso(),
                rebased=outcome.success,
                detail=outcome.detail,
            ),
        )
        _logger.info(
            "ralph runner terminal: run_id=%s node_id=%d success=%s detail=%s",
            request.run_id,
            request.node_id,
            outcome.success,
            outcome.detail,
        )
        return 0 if outcome.success else 1
    except Exception as exc:
        _logger.exception(
            "ralph runner failed: run_id=%s node_id=%d",
            request.run_id,
            request.node_id,
        )
        graph.finish_run(
            request.run_id,
            RunResult(
                status="failed",
                exit_code=1,
                timed_out=False,
                ended_at=now_iso(),
                rebased=False,
                detail=f"{type(exc).__name__}: {exc}",
            ),
        )
        return 1
    finally:
        graph.close()
