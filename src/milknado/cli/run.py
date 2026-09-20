"""run and attach commands — thin I/O parsing over milknado.app.run."""

from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, NamedTuple

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
from milknado.domains.graph import ControllerAuthorizationError

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
AttachedWatchOption = Annotated[
    bool,
    typer_option("--attached", help="Enable owner-fenced session input; read-only by default."),
]
WebOption = Annotated[bool, typer_option("--web", help="Serve the owner web view")]
RunPortOption = Annotated[int, typer_option("--port", min=1, max=65535, help="HTTP port")]
NoOpenOption = Annotated[bool, typer_option("--no-open", help="Do not open a browser")]


def _print_run_result(result: RunLoopResult) -> None:
    if result.root_done:
        console.print("[green]All nodes complete. Root goal achieved.[/green]")
    else:
        console.print(
            f"[yellow]Loop ended: {result.dispatched_total} dispatched, "
            + f"{result.completed_total} completed, "
            + f"{result.failed_total} failed.[/yellow]"
        )

    verification = result.verify_outcome
    if verification is not None and not verification.done:
        delta = verification.goal_delta or "no explanation provided"
        console.print(f"Verification incomplete: {delta}", markup=False, style="red")

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


def watch(
    project_root: Annotated[
        Path, typer_option("--project-root", help="Project root directory")
    ] = DEFAULT_PROJECT_ROOT,
    attached: AttachedWatchOption = False,
) -> None:
    """Observe durable run state; enable owner-fenced input only with --attached."""
    if not _is_interactive_terminal():
        console.print("[red]milknado watch requires an interactive terminal.[/red]")
        raise typer.Exit(code=2)

    project_root = project_root.resolve()
    config, plugins = _load_or_default(project_root)
    graph = None
    try:
        if attached:
            from milknado.app.watch import (
                AttachedWatchSource,
                WatchSnapshotSource,
                graph_command_admitter,
            )
            from milknado.app.watch_tui import run_attached_watch_tui

            graph = _ensure_db(config, plugins)
            source = AttachedWatchSource(
                WatchSnapshotSource(project_root, config.db_path), graph_command_admitter(graph)
            )
            run_attached_watch_tui(source)
        else:
            from milknado.app.watch_tui import run_watch_tui

            run_watch_tui(project_root, config.db_path)
    except (OSError, sqlite3.Error) as exc:
        console.print(f"[red]Cannot watch {project_root}: {exc}[/red]")
        raise typer.Exit(code=1) from None
    finally:
        if graph is not None:
            graph.close()


class RunCommandOptions(NamedTuple):
    project_root: Path
    strict: bool
    allow_protected: bool
    web: bool
    port: int
    no_open: bool


def run(  # noqa: PLR0913 - Typer requires one parameter per CLI option at this boundary.
    project_root: Annotated[
        Path, typer_option("--project-root", help="Project root directory")
    ] = DEFAULT_PROJECT_ROOT,
    strict: StrictOption = False,
    allow_protected: AllowProtectedOption = False,
    web: WebOption = False,
    port: RunPortOption = 8000,
    no_open: NoOpenOption = False,
) -> None:
    """Execute ready leaf nodes as parallel ralph loops."""
    _run(RunCommandOptions(project_root, strict, allow_protected, web, port, no_open))


def _run(options: RunCommandOptions) -> None:
    project_root, strict, allow_protected, web, port, no_open = options
    from milknado.app.run import (
        ProtectedBranchRefusal,
        build_execution_controller,
        ensure_dispatch_allowed,
        resolve_feature_branch,
        run_execution_loop,
    )
    from milknado.app.run_tui import run_execution_tui
    from milknado.cli._helpers import apply_runnable_root_exclusions
    from milknado.cli.web import OwnerWebContext, OwnerWebOptions, run_owner_web
    from milknado.domains.dispatch import reconcile_orphaned_runs
    from milknado.domains.execution import get_dispatchable_nodes

    project_root = project_root.resolve()
    config, plugins = _load_or_default(project_root)
    graph = None

    try:
        if web:
            result = run_owner_web(
                OwnerWebContext(project_root, config, plugins),
                OwnerWebOptions(
                    strict=strict,
                    allow_protected=allow_protected,
                    port=port,
                    no_open=no_open,
                ),
            )
            if result is None:
                return
            _print_run_result(result)
            if result.strict_exit:
                raise typer.Exit(code=1)
            return

        feature_branch = resolve_feature_branch(project_root)
        ensure_dispatch_allowed(config, feature_branch, allow_protected)
        graph = _ensure_db(config, plugins)

        _ = graph.reconcile_completed_goals()
        exclusions = apply_runnable_root_exclusions(graph, console)

        interactive = _is_interactive_terminal()
        if not interactive:
            _ = reconcile_orphaned_runs(graph)
        controller = (
            build_execution_controller(graph, config, project_root) if interactive else None
        )
        dispatchable = get_dispatchable_nodes(graph)
        if not any(node not in exclusions.excluded for node in dispatchable):
            console.print("No nodes ready for execution.")
            if exclusions.has_errors:
                raise typer.Exit(code=1)
            return

        console.print(f"Starting execution loop on [bold]{feature_branch}[/bold]...")
        if interactive:
            assert controller is not None
            result = run_execution_tui(
                controller,
                feature_branch=feature_branch,
                strict=strict,
                allow_protected=allow_protected,
            )
            if result is None:
                return
        else:
            result = run_execution_loop(
                graph,
                config,
                project_root,
                feature_branch,
                strict,
                allow_protected=allow_protected,
            )

        _print_run_result(result)
        if result.strict_exit:
            raise typer.Exit(code=1)
    except ProtectedBranchRefusal as refusal:
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
        raise typer.Exit(code=2) from None
    except ControllerAuthorizationError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from None
    finally:
        if graph is not None:
            graph.close()
