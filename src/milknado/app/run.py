"""Application-layer policy and adapter wiring for the run / dispatch surfaces.

The CLI ``run``/``attach`` commands and the MCP ``milknado_run_inline*`` tools are
thin: they parse I/O and call the functions here, which own the policy (protected
branch refusal, execution-config assembly, worker-cmd validation, worktree
isolation) and construct the adapters (git, loop, crg, process, tmux). Entry
modules therefore hold no inline dispatch policy and build no adapters.
"""

from __future__ import annotations

import logging
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import typer
from rich.console import Console

from milknado.adapters import ProcessAdapter, TmuxAdapter
from milknado.domains.common import (
    MilknadoConfig,
    NodeKind,
    NodeStatus,
    WorktreeMode,
    resolve_flavor_profile,
)

if TYPE_CHECKING:
    from milknado.domains.execution import ExecutionConfig
    from milknado.domains.execution.run_loop import RunLoopResult
    from milknado.domains.graph import MikadoGraph

console = Console()
_logger = logging.getLogger(__name__)


def check_protected_branch(
    cfg: MilknadoConfig,
    branch: str,
    allow_protected: bool,
) -> None:
    """Refuse to run on a protected or invalid branch with exit code 2.

    Called before the graph DB, executor, or run log is built, so a refused run
    leaves no side effects (US-102.2). Refusal is loud — the user is told which
    branch was refused and how to override. ``--allow-protected`` is the explicit
    opt-in for a protected *named* branch; a detached HEAD (``current_branch()``
    returns ``"HEAD"`` or ``""``) is refused unconditionally, since the run loop
    has no valid branch to rebase-merge completed nodes back onto (mirrors the
    headless guard in ``execution/headless.py``).
    """
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


def build_exec_config(config: MilknadoConfig, project_root: Path) -> ExecutionConfig:
    from milknado.domains.execution import ExecutionConfig

    return ExecutionConfig(
        execution_agent=config.execution_agent,
        quality_gates=config.quality_gates,
        worktree_pattern=config.worktree_pattern,
        project_root=project_root,
    )


def resolve_feature_branch(project_root: Path) -> str:
    """Return the checkout's current branch name (adapter wiring for the CLI)."""
    from milknado.adapters import GitAdapter

    return GitAdapter(project_root).current_branch()


def run_execution_loop(
    graph: MikadoGraph,
    config: MilknadoConfig,
    project_root: Path,
    feature_branch: str,
    strict: bool,
) -> RunLoopResult:
    """Wire the executor + run loop and drive it to completion.

    Owns the adapter composition (git, loop, crg, executor, run loop) so the CLI
    ``run`` command never constructs an adapter or holds this policy inline.
    """
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


def resolve_run_attach_target(graph: MikadoGraph, project_root: Path, run_id: str) -> str:
    """Resolve the tmux window target for a run (adapter wiring for ``attach``)."""
    from milknado.domains.dispatch import resolve_attach_target

    return resolve_attach_target(graph, TmuxAdapter(project_root), run_id)


def validate_worker_cmd(worker_cmd: str | None) -> None:
    """Reject an explicit worker_cmd whose executable isn't an allowed AI agent CLI.

    Eager pre-check on the MCP arg; the env fallback and built-in default are
    validated again where they're resolved (``runner._resolve_worker_cmd``). Both
    routes share ``validate_worker_argv``, so the allowlist lives in one place.
    """
    from milknado.domains.dispatch import validate_worker_argv

    if not worker_cmd or not worker_cmd.strip():
        return
    validate_worker_argv(shlex.split(worker_cmd))


def prepare_isolation(
    graph,  # noqa: ANN001
    git,  # noqa: ANN001
    root: Path,
    node,  # noqa: ANN001
    run_id: str,
    worktree: WorktreeMode,
    merge_back: bool,
    worktree_pattern: str,
):  # noqa: ANN201
    from milknado.domains.dispatch import setup_isolated_worktree

    if worktree != WorktreeMode.ISOLATE:
        return root, None
    context = setup_isolated_worktree(graph, git, root, node, run_id, worktree_pattern)
    return context.worktree_path, context if merge_back else None


@dataclass(frozen=True)
class InlineRunRequest:
    node_id: int
    worker_cmd: str | None
    timeout_seconds: int
    worktree: WorktreeMode
    merge_back: bool


def _require_task_node(graph, node_id: int):  # noqa: ANN001, ANN202
    node = graph.get_node(node_id)
    if node is None:
        raise ValueError(f"node {node_id} not found")
    if node.kind != NodeKind.TASK:
        raise ValueError(
            f"node {node_id} has kind={node.kind.value}; only task nodes can be dispatched"
        )
    return node


def run_inline(graph, cfg, root: Path, request: InlineRunRequest) -> dict:  # noqa: ANN001
    """Dispatch a node to a blocking subprocess worker; return the run state dict."""
    from milknado.adapters import GitAdapter
    from milknado.domains.dispatch import SyncDispatchRequest, dispatch_node_sync

    _logger.info(
        "milknado_run_inline: node=%d timeout=%ds worktree=%s merge_back=%s",
        request.node_id,
        request.timeout_seconds,
        request.worktree.value,
        request.merge_back,
    )
    node = _require_task_node(graph, request.node_id)
    profile = resolve_flavor_profile(cfg, node.flavor)
    state = dispatch_node_sync(
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
    return state if isinstance(state, dict) else vars(state)


def run_inline_start(graph, cfg, root: Path, request: InlineRunRequest, use_tmux: bool) -> dict:  # noqa: ANN001
    """Start an async worker (optionally in tmux); return the initial run state dict."""
    from milknado.adapters import GitAdapter
    from milknado.app.project import open_graph
    from milknado.domains.dispatch import (
        AsyncRunRequest,
        GraphSessionPort,
        ensure_tmux_ready,
        make_run_id,
        now_iso,
        reclaim_stale_node,
        render_brief,
        start_headless_async,
    )

    class _GraphSessions(GraphSessionPort):
        def open_graph(self, project_root: Path):  # noqa: ANN201
            return open_graph(project_root)

    tmux: TmuxAdapter | None = None
    if use_tmux:
        # Fail closed BEFORE any claim: tmux was explicitly requested, so a
        # missing binary or unstartable server fails the dispatch loudly.
        tmux = TmuxAdapter(root)
        ensure_tmux_ready(tmux)
    git = GitAdapter(root)
    node = _require_task_node(graph, request.node_id)
    run_id = make_run_id(request.node_id)
    if node.status == NodeStatus.RUNNING:
        reclaim_stale_node(graph, request.node_id, fence_run_id=node.run_id)
    profile = resolve_flavor_profile(cfg, node.flavor)
    brief = render_brief(graph, request.node_id, prepend=profile.brief_prepend)
    graph.claim_node_for_dispatch(request.node_id, run_id, now=now_iso())
    try:
        worker_cwd, merge_ctx = prepare_isolation(
            graph,
            git,
            root,
            node,
            run_id,
            request.worktree,
            request.merge_back,
            cfg.worktree_pattern,
        )
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
        # Startup failed after the claim: release the claim with a fenced terminal
        # write so the node is not stranded RUNNING, then re-raise.
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
