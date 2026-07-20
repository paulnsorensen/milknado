"""Application-layer policy for run commands: inline dispatch, exec config, branch guard."""

from __future__ import annotations

import logging
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from milknado.domains.execution import ExecutionConfig
    from milknado.domains.execution.run_loop import RunLoopResult

import typer
from rich.console import Console

from milknado.adapters import TmuxAdapter
from milknado.domains.common import MilknadoConfig, WorktreeMode

console = Console()
_logger = logging.getLogger(__name__)


def check_protected_branch(
    cfg: MilknadoConfig,
    branch: str,
    allow_protected: bool,
) -> None:
    """Refuse to run on a protected or invalid branch with exit code 2."""
    if branch in ("", "HEAD"):
        console.print(
            f"[red]Refusing to run on detached HEAD (branch {branch!r}); "
            "check out a named branch first.[/red]"
        )
        raise typer.Exit(code=2)
    if not allow_protected and branch in cfg.protected_branches:
        console.print(
            f"[red]Refusing to run on protected branch '{branch}'. "
            "Pass --allow-protected to override.[/red]"
        )
        raise typer.Exit(code=2)


def build_exec_config(
    config: MilknadoConfig,
    project_root: Path,
) -> ExecutionConfig:
    from milknado.domains.execution import ExecutionConfig

    return ExecutionConfig(
        execution_agent=config.execution_agent,
        quality_gates=config.quality_gates,
        worktree_pattern=config.worktree_pattern,
        project_root=project_root,
    )


def validate_worker_cmd(worker_cmd: str | None) -> None:
    """Reject an explicit worker_cmd whose executable is not an allowed AI agent CLI."""
    from milknado.domains.dispatch import validate_worker_argv

    if not worker_cmd or not worker_cmd.strip():
        return
    validate_worker_argv(shlex.split(worker_cmd))


def resolve_feature_branch(project_root: Path) -> str:
    """Return the current git branch name for the project root."""
    from milknado.adapters import GitAdapter

    return GitAdapter(project_root).current_branch()


def resolve_run_attach_target(graph, project_root: Path, run_id: str) -> str:  # noqa: ANN001
    """Resolve the tmux attach target for a run ID."""
    from milknado.domains.dispatch import resolve_attach_target

    return resolve_attach_target(graph, TmuxAdapter(project_root), run_id)


def run_execution_loop(
    graph,  # noqa: ANN001
    config: MilknadoConfig,
    project_root: Path,
    feature_branch: str,
    strict: bool,
) -> RunLoopResult:
    """Build adapters and run the execution loop to completion."""
    from milknado.adapters import CrgAdapter, GitAdapter, LoopAdapter
    from milknado.domains.execution import Executor, RunLoop

    git = GitAdapter(project_root)
    ralph = LoopAdapter()
    crg = CrgAdapter(project_root)
    executor = Executor(graph=graph, git=git, ralph=ralph, crg=crg)
    loop = RunLoop(executor=executor, graph=graph, ralph=ralph, config=config)
    return loop.run(
        config=build_exec_config(config, project_root),
        feature_branch=feature_branch,
        concurrency_limit=config.concurrency_limit,
        strict=strict,
    )


@dataclass(frozen=True)
class InlineRunRequest:
    node_id: int
    worker_cmd: str | None
    timeout_seconds: int
    worktree: WorktreeMode
    merge_back: bool


def run_inline(
    graph,  # noqa: ANN001
    cfg: MilknadoConfig,
    root: Path,
    request: InlineRunRequest,
) -> object:
    """Run a node inline synchronously; returns a run-state dict."""
    from milknado.adapters import GitAdapter, ProcessAdapter
    from milknado.domains.common import NodeKind, resolve_flavor_profile
    from milknado.domains.dispatch import SyncDispatchRequest, dispatch_node_sync

    node = graph.get_node(request.node_id)
    if node is None:
        raise ValueError(f"node {request.node_id} not found")
    if node.kind != NodeKind.TASK:
        raise ValueError(
            f"node {request.node_id} has kind={node.kind.value}; only task nodes can be dispatched"
        )
    profile = resolve_flavor_profile(cfg, node.flavor)
    return dispatch_node_sync(
        graph,
        GitAdapter(root),
        SyncDispatchRequest(
            node_id=request.node_id,
            project_root=root,
            worker_cmd=request.worker_cmd,
            timeout_seconds=request.timeout_seconds,
            default_cmd=profile.execution_agent,
            process=ProcessAdapter(),
            brief_prepend=profile.brief_prepend,
            worktree_mode=request.worktree,
            merge_back=request.merge_back,
            worktree_pattern=cfg.worktree_pattern,
        ),
    )


def run_inline_start(
    graph,  # noqa: ANN001
    cfg: MilknadoConfig,
    root: Path,
    request: InlineRunRequest,
    use_tmux: bool,
) -> object:
    """Start an async inline run; returns a run-state dict."""
    from milknado.adapters import GitAdapter, ProcessAdapter
    from milknado.app.project import open_graph as _open_graph
    from milknado.domains.common import NodeKind, NodeStatus, resolve_flavor_profile
    from milknado.domains.dispatch import (
        AsyncRunRequest,
        GraphSessionPort,
        ensure_tmux_ready,
        make_run_id,
        now_iso,
        reclaim_stale_node,
        render_brief,
        setup_isolated_worktree,
        start_headless_async,
    )

    class _GraphSessions(GraphSessionPort):
        def open_graph(self, project_root: Path):  # noqa: ANN201
            return _open_graph(project_root)

    tmux: TmuxAdapter | None = None
    if use_tmux:
        tmux = TmuxAdapter(root)
        ensure_tmux_ready(tmux)

    git = GitAdapter(root)
    node = graph.get_node(request.node_id)
    if node is None:
        raise ValueError(f"node {request.node_id} not found")
    if node.kind != NodeKind.TASK:
        raise ValueError(
            f"node {request.node_id} has kind={node.kind.value}; only task nodes can be dispatched"
        )
    run_id = make_run_id(request.node_id)
    if node.status == NodeStatus.RUNNING:
        reclaim_stale_node(graph, request.node_id, fence_run_id=node.run_id)
    profile = resolve_flavor_profile(cfg, node.flavor)
    brief = render_brief(graph, request.node_id, prepend=profile.brief_prepend)
    graph.claim_node_for_dispatch(request.node_id, run_id, now=now_iso())
    worker_cwd = root
    merge_ctx = None
    if request.worktree == WorktreeMode.ISOLATE:
        context = setup_isolated_worktree(graph, git, root, node, run_id, cfg.worktree_pattern)
        worker_cwd = context.worktree_path
        merge_ctx = context if request.merge_back else None
    try:
        ref = start_headless_async(
            AsyncRunRequest(
                project_root=root,
                node_id=request.node_id,
                brief=brief,
                worker_cmd=request.worker_cmd,
                timeout_seconds=request.timeout_seconds,
                run_id=run_id,
                default_cmd=profile.execution_agent,
                cwd=worker_cwd,
                merge_ctx=merge_ctx,
            ),
            _GraphSessions(),
            git,
            ProcessAdapter(),
            tmux,
        )
    except Exception:
        graph.mark_terminal(request.node_id, run_id, NodeStatus.FAILED)
        raise
    _logger.info(
        "milknado_run_inline_start: node=%d run_id=%s timeout=%ds",
        request.node_id,
        ref.run_id,
        request.timeout_seconds,
    )
    return {
        "run_id": ref.run_id,
        "node_id": request.node_id,
        "status": "running",
        "log_path": str(ref.log_path),
    }
