"""run command and execution helpers."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated

if TYPE_CHECKING:
    from milknado.domains.execution import ExecutionConfig
    from milknado.domains.execution.run_loop import RunLoopResult

import typer
from rich.console import Console

from milknado._cli_helpers import _ensure_db, _load_or_default
from milknado.domains.common import MilknadoConfig

console = Console()


def _build_exec_config(
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


def _print_run_result(result: RunLoopResult) -> None:
    if result.root_done:
        console.print("[green]All nodes complete. Root goal achieved.[/green]")
    else:
        console.print(
            f"[yellow]Loop ended: {result.completed_total} completed, "
            f"{result.failed_total} failed.[/yellow]"
        )

    for pr in result.stacked_prs:
        console.print(
            f"  [cyan]PR[/cyan] node {pr.node_id} ({pr.branch} → {pr.base_branch}): {pr.pr_url}"
        )

    for conflict in result.rebase_conflicts:
        console.print(
            f"\n[red bold]Rebase conflict — node {conflict.node_id}:[/red bold] "
            f"{conflict.description}",
        )
        if conflict.conflicting_files:
            for f in conflict.conflicting_files:
                console.print(f"  [red]•[/red] {f}")
        if conflict.detail:
            console.print(f"  [dim]{conflict.detail}[/dim]")


def run(
    project_root: Annotated[
        Path, typer.Option("--project-root", help="Project root directory")
    ] = Path("."),
    resume: Annotated[
        bool,
        typer.Option("--resume/--no-resume", help="Reconcile stale RUNNING nodes before run."),
    ] = False,
) -> None:
    """Execute ready leaf nodes as parallel ralph loops."""
    from milknado.adapters import CrgAdapter, GitAdapter, RalphifyAdapter
    from milknado.domains.execution import (
        Executor,
        RunLoop,
        get_dispatchable_nodes,
        reconcile_stale_nodes,
    )

    project_root = project_root.resolve()
    config, plugins = _load_or_default(project_root)
    graph = _ensure_db(config, plugins)

    try:
        git = GitAdapter(project_root)
        ralph = RalphifyAdapter()
        feature_branch = git.current_branch()

        if resume:
            result = reconcile_stale_nodes(graph, git, ralph, feature_branch)
            if result.resolved_done or result.reset_pending:
                console.print(
                    f"[dim]Reconciled: {len(result.resolved_done)} marked done, "
                    f"{len(result.reset_pending)} reset to pending.[/dim]"
                )

        if not get_dispatchable_nodes(graph):
            console.print("No nodes ready for execution.")
            return

        crg = CrgAdapter(project_root)
        executor = Executor(graph=graph, git=git, ralph=ralph, crg=crg)
        loop = RunLoop(executor=executor, graph=graph, ralph=ralph)
        console.print(f"Starting execution loop on [bold]{feature_branch}[/bold]...")
        run_result = loop.run(
            config=_build_exec_config(config, project_root),
            feature_branch=feature_branch,
            concurrency_limit=config.concurrency_limit,
            pr_stack=config.pr_stack,
        )
        _print_run_result(run_result)
    finally:
        graph.close()
