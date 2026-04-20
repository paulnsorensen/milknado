from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from milknado.app.command_ops import (
    execute_add_node,
    execute_agents_check,
    execute_crg,
    execute_doctor,
    execute_index,
    execute_init,
    execute_status,
)
from milknado.app.run_command import execute_plan, execute_run
from milknado.app.spec_ingest import SpecIngestError, validate_spec_paths
from milknado.domains.common.toolchain import (
    get_required_tool_status,
    install_missing_rust_tools,
)

app = typer.Typer(name="milknado", help="Mikado execution engine")
console = Console()


def _flatten_csv(items: list[str] | None) -> list[str]:
    return [p.strip() for raw in (items or []) for p in raw.split(",") if p.strip()]


def _print_tool_status() -> list[tuple[str, bool]]:
    statuses = get_required_tool_status()
    for s in statuses:
        state = "ok" if s.installed else "missing"
        console.print(f"{s.name}: {state}{f' ({s.path})' if s.path else ''}")
    return [(s.name, s.installed) for s in statuses]


def _install_rust_tools_or_exit() -> None:
    installed, failed = install_missing_rust_tools()
    for name in installed:
        console.print(f"[green]Installed {name}[/green]")
    if failed:
        console.print("[red]Failed to install:[/red] " + ", ".join(failed))
        console.print("Install Rust/cargo, then run: [bold]milknado tools install[/bold]")
        raise typer.Exit(code=1)


@app.command()
def init(
    project_root: Annotated[Path, typer.Argument(help="Project root directory")] = Path("."),
    install_rust_tools: Annotated[
        bool, typer.Option("--install-rust-tools", help="Install missing Rust tools via cargo.")
    ] = False,
) -> None:
    """Initialize milknado in a project directory."""
    execute_init(project_root.resolve())
    if install_rust_tools:
        _install_rust_tools_or_exit()


@app.command()
def index(
    project_root: Annotated[Path, typer.Argument(help="Project root directory")] = Path("."),
) -> None:
    """Rebuild the code-review-graph index."""
    execute_index(project_root.resolve())


@app.command()
def status(
    project_root: Annotated[Path, typer.Argument(help="Project root directory")] = Path("."),
    with_durations: Annotated[
        bool, typer.Option("--with-durations", help="Print per-node duration column and summary.")
    ] = False,
) -> None:
    """Show the current state of the Mikado graph."""
    execute_status(project_root.resolve(), with_durations)


@app.command()
def crg(
    project_root: Annotated[Path, typer.Argument(help="Project root directory")] = Path("."),
) -> None:
    """Show the code-review-graph architecture overview."""
    execute_crg(project_root.resolve())


@app.command("add-node")
def add_node(
    description: Annotated[str, typer.Argument(help="Node description")],
    parent: Annotated[
        int | None, typer.Option("--parent", "-p", help="Parent node ID")
    ] = None,
    files: Annotated[
        list[str] | None, typer.Option("--files", "-f", help="Files this node will touch")
    ] = None,
    project_root: Annotated[
        Path, typer.Option("--project-root", help="Project root directory")
    ] = Path("."),
) -> None:
    """Add a node to the Mikado graph."""
    execute_add_node(project_root.resolve(), description, parent, files)


@app.command()
def plan(
    spec: Annotated[
        list[str] | None, typer.Option(help="Spec .md files. Repeat or comma-separate.")
    ] = None,
    issue: Annotated[
        list[str] | None, typer.Option(help="GitHub issue ref. Repeat or comma-separate.")
    ] = None,
    project_root: Annotated[
        Path, typer.Option("--project-root", help="Project root directory")
    ] = Path("."),
    max_verify_rounds: Annotated[
        int, typer.Option("--max-verify-rounds", help="Max verify-spec rounds (0=disabled).")
    ] = 3,
    resume: Annotated[
        bool, typer.Option(help="Append to existing plan without dropping nodes.")
    ] = False,
    reset: Annotated[
        bool, typer.Option(help="Drop existing plan and re-plan from scratch.")
    ] = False,
    mega_batch_threshold: Annotated[
        int,
        typer.Option("--mega-batch-threshold", help="Abort if batch exceeds this (0=disabled)."),
    ] = 5,
    force_single_batch: Annotated[
        bool, typer.Option("--force-single-batch", help="Skip mega-batch guard.")
    ] = False,
) -> None:
    """Launch interactive planning session to decompose one or more specs/issues."""
    if resume and reset:
        console.print("[red]--resume and --reset are mutually exclusive.[/red]")
        raise typer.Exit(code=1)
    try:
        spec_paths = validate_spec_paths(_flatten_csv(spec))
    except SpecIngestError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=1) from None
    issue_refs = _flatten_csv(issue)
    if not spec_paths and not issue_refs:
        console.print("[red]Provide --spec or --issue.[/red]")
        raise typer.Exit(code=1)
    execute_plan(
        project_root.resolve(), spec_paths, issue_refs,
        max_verify_rounds, resume, reset, mega_batch_threshold, force_single_batch,
    )


@app.command()
def run(
    project_root: Annotated[
        Path, typer.Option("--project-root", help="Project root directory")
    ] = Path("."),
    strict: Annotated[
        bool, typer.Option(help="Drain active workers then exit 1 on first failure.")
    ] = False,
    allow_protected: Annotated[
        bool, typer.Option("--allow-protected", help="Bypass protected-branch guard.")
    ] = False,
    spec: Annotated[
        Path | None,
        typer.Option(
            exists=True, file_okay=True, dir_okay=False, readable=True, resolve_path=True,
            help="Spec .md file; enables verify-spec and replan-on-gaps after run.",
        ),
    ] = None,
    max_verify_rounds: Annotated[
        int, typer.Option("--max-verify-rounds", help="Max verify-spec rounds (0=disabled).")
    ] = 3,
) -> None:
    """Execute ready leaf nodes as parallel ralph loops."""
    execute_run(
        project_root.resolve(), strict, allow_protected, spec, max_verify_rounds,
    )


@app.command()
def doctor(
    project_root: Annotated[Path, typer.Argument(help="Project root directory")] = Path("."),
) -> None:
    """Run health checks on the milknado installation."""
    execute_doctor(project_root.resolve())


agents_app = typer.Typer(name="agents", help="Agent CLI compatibility")
app.add_typer(agents_app)


@agents_app.command("check")
def agents_check(
    project_root: Annotated[
        Path, typer.Option("--project-root", help="Project root directory")
    ] = Path("."),
) -> None:
    """Print resolved planning/execution commands and sample planning argv."""
    execute_agents_check(project_root.resolve())


plugin_app = typer.Typer(name="plugin", help="Plugin management commands")
app.add_typer(plugin_app)

tools_app = typer.Typer(name="tools", help="Toolchain commands")
app.add_typer(tools_app)


@tools_app.command("check")
def tools_check() -> None:
    """Check whether required Rust tools are available."""
    if not all(installed for _, installed in _print_tool_status()):
        raise typer.Exit(code=1)


@tools_app.command("install")
def tools_install() -> None:
    """Install missing Rust tools used by milknado workflows."""
    _install_rust_tools_or_exit()
    _print_tool_status()


@plugin_app.command("init")
def plugin_init(
    name: Annotated[str, typer.Argument(help="Plugin name")],
    target_dir: Annotated[
        Path, typer.Option("--target-dir", "-d", help="Directory to create plugin in")
    ] = Path("."),
) -> None:
    """Scaffold a new milknado plugin."""
    from milknado.plugins import scaffold_plugin

    try:
        result = scaffold_plugin(name, target_dir.resolve())
        console.print(f"Created plugin [bold]{name}[/bold] at {result.plugin_dir}")
        for f in result.files_created:
            console.print(f"  {f}")
    except FileExistsError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=1) from None
