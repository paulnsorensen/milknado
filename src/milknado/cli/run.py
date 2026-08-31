"""run and attach commands — thin I/O parsing over milknado.app.run."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

if TYPE_CHECKING:
    from milknado.domains.execution import RunLoopResult

import typer
from rich.console import Console

from milknado.cli._helpers import (
    DEFAULT_PROJECT_ROOT,
    typer_argument,
    typer_option,
)
from milknado.cli._helpers import (
    ensure_db as _ensure_db,
)
from milknado.cli._helpers import (
    load_or_default as _load_or_default,
)

console = Console()

StrictOption = Annotated[
    bool,
    typer_option(
        "--strict",
        help="Exit 1 if any node fails mid-run (drain in-flight, no new dispatch).",
    ),
]
AllowProtectedOption = Annotated[
    bool,
    typer_option(
        "--allow-protected",
        help="Permit execution on a protected branch (e.g. main).",
    ),
]


def _print_run_result(result: RunLoopResult) -> None:
    if result.root_done:
        console.print("[green]All nodes complete. Root goal achieved.[/green]")
    else:
        console.print(
            f"[yellow]Loop ended: {result.dispatched_total} dispatched, "
            + f"{result.completed_total} completed, "
            + f"{result.failed_total} failed.[/yellow]"
        )

    for conflict in result.rebase_conflicts:
        console.print(
            f"\n[red bold]Rebase conflict — node {conflict.node_id}:[/red bold] "
            + f"{conflict.description}",
        )
        if conflict.conflicting_files:
            for f in conflict.conflicting_files:
                console.print(f"  [red]•[/red] {f}")
        if conflict.detail:
            console.print(f"  [dim]{conflict.detail}[/dim]")


def _tmux_attach_argv(target: str) -> list[str]:
    """The tmux client argv that focuses the run's window and attaches.

    Inside an existing tmux client a nested attach is refused, so the current
    client is switched instead.
    """
    session = target.split(":", 1)[0]
    follow = "switch-client" if os.environ.get("TMUX") else "attach-session"
    return ["tmux", "select-window", "-t", target, ";", follow, "-t", session]


def attach(
    run_id: Annotated[str, typer_argument(help="Run id, as returned by the run-start tools")],
    project_root: Annotated[
        Path, typer_option("--project-root", help="Project root directory")
    ] = DEFAULT_PROJECT_ROOT,
) -> None:
    """Attach to a running tmux-dispatched run's window."""
    from milknado.app.run import resolve_run_attach_target

    project_root = project_root.resolve()
    config, plugins = _load_or_default(project_root)
    graph = _ensure_db(config, plugins)
    try:
        target = resolve_run_attach_target(graph, project_root, run_id)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None
    finally:
        graph.close()
    argv = _tmux_attach_argv(target)
    os.execvp(argv[0], argv)


def _is_interactive_terminal() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def run(
    project_root: Annotated[
        Path, typer_option("--project-root", help="Project root directory")
    ] = DEFAULT_PROJECT_ROOT,
    strict: StrictOption = False,
    allow_protected: AllowProtectedOption = False,
) -> None:
    """Execute ready leaf nodes as parallel ralph loops."""
    from milknado.app.run import (
        build_execution_controller,
        check_protected_branch,
        resolve_feature_branch,
        run_execution_loop,
    )
    from milknado.app.run_tui import run_execution_tui
    from milknado.domains.dispatch import reconcile_orphaned_runs
    from milknado.domains.execution import get_dispatchable_nodes
    from milknado.domains.graph import invalid_subtree_node_ids, validate_runnable_roots

    project_root = project_root.resolve()
    config, plugins = _load_or_default(project_root)

    feature_branch = resolve_feature_branch(project_root)
    refusal = check_protected_branch(config, feature_branch, allow_protected)
    if refusal is not None:
        if refusal.reason == "detached":
            console.print(
                f"[red]Refusing to run on detached HEAD (branch {refusal.branch!r}); "
                + "check out a named branch first.[/red]"
            )
        else:
            console.print(
                f"[red]Refusing to run on protected branch '{refusal.branch}'. "
                + "Pass --allow-protected to override.[/red]"
            )
        raise typer.Exit(code=2)

    graph = _ensure_db(config, plugins)

    try:
        _ = graph.reconcile_completed_goals()
        root_reports = validate_runnable_roots(graph)
        excluded: set[int] = set()
        has_errors = False
        for goal_id, report in root_reports:
            for msg in report.warnings:
                console.print(f"[yellow]warning (goal {goal_id}): {msg}[/yellow]")
            if report.errors:
                has_errors = True
                excluded.update(invalid_subtree_node_ids(graph, goal_id))
                for msg in report.errors:
                    console.print(f"[yellow]warning (goal {goal_id}; skipped): {msg}[/yellow]")
        graph.set_dispatch_exclusions(excluded)

        interactive = _is_interactive_terminal()
        if not interactive:
            _ = reconcile_orphaned_runs(graph)
        controller = (
            build_execution_controller(graph, config, project_root) if interactive else None
        )
        if not any(node not in excluded for node in get_dispatchable_nodes(graph)):
            console.print("No nodes ready for execution.")
            if has_errors:
                raise typer.Exit(code=1)
            return

        console.print(f"Starting execution loop on [bold]{feature_branch}[/bold]...")
        if interactive:
            assert controller is not None
            result = run_execution_tui(
                controller,
                feature_branch=feature_branch,
                strict=strict,
            )
            if result is None:
                return
        else:
            result = run_execution_loop(graph, config, project_root, feature_branch, strict)
        _print_run_result(result)
        if result.strict_exit:
            raise typer.Exit(code=1)
    finally:
        graph.close()
