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

StrictOption = Annotated[
    bool,
    typer.Option(
        "--strict",
        help="Exit 1 if any node fails mid-run (drain in-flight, no new dispatch).",
    ),
]
AllowProtectedOption = Annotated[
    bool,
    typer.Option(
        "--allow-protected",
        help="Permit execution on a protected branch (e.g. main).",
    ),
]


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
    strict: StrictOption = False,
    allow_protected: AllowProtectedOption = False,
) -> None:
    """Execute ready leaf nodes as parallel ralph loops."""
    from milknado.adapters import CrgAdapter, GitAdapter, RalphifyAdapter
    from milknado.app.run_command import check_protected_branch
    from milknado.domains.execution import Executor, RunLoop, get_dispatchable_nodes
    from milknado.domains.graph.runnability import validate_runnable_roots

    project_root = project_root.resolve()
    config, plugins = _load_or_default(project_root)

    git = GitAdapter(project_root)
    feature_branch = git.current_branch()
    check_protected_branch(config, feature_branch, allow_protected)

    graph = _ensure_db(config, plugins)

    try:
        root_reports = validate_runnable_roots(graph)
        all_errors: list[str] = []
        for goal_id, report in root_reports:
            for msg in report.warnings:
                console.print(f"[yellow]warning (goal {goal_id}): {msg}[/yellow]")
            all_errors.extend(f"goal {goal_id}: {e}" for e in report.errors)
        if all_errors:
            for msg in all_errors:
                console.print(f"[red]error: {msg}[/red]")
            raise typer.Exit(code=1)

        if not get_dispatchable_nodes(graph):
            console.print("No nodes ready for execution.")
            return

        ralph = RalphifyAdapter()
        crg = CrgAdapter(project_root)
        executor = Executor(graph=graph, git=git, ralph=ralph, crg=crg)
        loop = RunLoop(executor=executor, graph=graph, ralph=ralph)
        console.print(f"Starting execution loop on [bold]{feature_branch}[/bold]...")
        result = loop.run(
            config=_build_exec_config(config, project_root),
            feature_branch=feature_branch,
            concurrency_limit=config.concurrency_limit,
            strict=strict,
        )
        _print_run_result(result)
        if result.strict_exit:
            raise typer.Exit(code=1)
    finally:
        graph.close()
