from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from milknado.domains.common import (
    MilknadoConfig,
    default_config,
    load_config,
    save_config,
)
from milknado.domains.graph import MikadoGraph

app = typer.Typer(name="milknado", help="Mikado execution engine")
console = Console()

CONFIG_FILE = "milknado.toml"


def _find_config(project_root: Path) -> Path:
    return project_root / CONFIG_FILE


def _load_or_default(project_root: Path) -> MilknadoConfig:
    config_path = _find_config(project_root)
    if config_path.exists():
        return load_config(config_path)
    return default_config(project_root)


def _ensure_db(config: MilknadoConfig) -> MikadoGraph:
    config.db_path.parent.mkdir(parents=True, exist_ok=True)
    return MikadoGraph(config.db_path)


@app.command()
def init(
    project_root: Annotated[
        Path, typer.Argument(help="Project root directory")
    ] = Path("."),
) -> None:
    """Initialize milknado in a project directory."""
    project_root = project_root.resolve()
    config_path = _find_config(project_root)

    if config_path.exists():
        console.print(f"Config already exists: {config_path}")
        config = load_config(config_path)
    else:
        config = default_config(project_root)
        save_config(config, config_path)
        console.print(f"Created config: {config_path}")

    graph = _ensure_db(config)
    graph.close()
    console.print(f"Database ready: {config.db_path}")


@app.command()
def status(
    project_root: Annotated[
        Path, typer.Argument(help="Project root directory")
    ] = Path("."),
) -> None:
    """Show the current state of the Mikado graph."""
    project_root = project_root.resolve()
    config = _load_or_default(project_root)
    graph = _ensure_db(config)

    try:
        nodes = graph.get_all_nodes()
        if not nodes:
            console.print("No nodes in graph. Run [bold]milknado plan[/bold] to start.")
            return

        done = sum(1 for n in nodes if n.status.value == "done")
        total = len(nodes)
        pct = (done / total * 100) if total else 0

        console.print(f"\n[bold]Progress:[/bold] {done}/{total} ({pct:.0f}%)\n")

        ready = graph.get_ready_nodes()
        ready_ids = {n.id for n in ready}

        table = Table(title="Mikado Graph")
        table.add_column("ID", style="dim")
        table.add_column("Description")
        table.add_column("Status")
        table.add_column("Ready")

        status_colors = {
            "pending": "yellow",
            "running": "blue",
            "done": "green",
            "blocked": "red",
            "failed": "red bold",
        }

        for node in nodes:
            color = status_colors.get(node.status.value, "white")
            is_ready = "✓" if node.id in ready_ids else ""
            table.add_row(
                str(node.id),
                node.description,
                f"[{color}]{node.status.value}[/{color}]",
                is_ready,
            )

        console.print(table)
    finally:
        graph.close()


@app.command("add-node")
def add_node(
    description: Annotated[str, typer.Argument(help="Node description")],
    parent: Annotated[
        int | None, typer.Option("--parent", "-p", help="Parent node ID")
    ] = None,
    files: Annotated[
        list[str] | None,
        typer.Option("--files", "-f", help="Files this node will touch"),
    ] = None,
    project_root: Annotated[
        Path, typer.Option("--project-root", help="Project root directory")
    ] = Path("."),
) -> None:
    """Add a node to the Mikado graph."""
    project_root = project_root.resolve()
    config = _load_or_default(project_root)
    graph = _ensure_db(config)

    try:
        node = graph.add_node(description, parent_id=parent)
        if files:
            graph.set_file_ownership(node.id, files)

        if parent is not None:
            parent_node = graph.get_node(parent)
            if parent_node and parent_node.status.value == "running":
                graph.mark_blocked(parent)
                console.print(f"Parent node {parent} marked as blocked.")

        console.print(f"Added node {node.id}: {node.description}")
    finally:
        graph.close()


@app.command()
def plan(
    goal: Annotated[str, typer.Argument(help="Goal to decompose")],
    project_root: Annotated[
        Path, typer.Option("--project-root", help="Project root directory")
    ] = Path("."),
) -> None:
    """Launch interactive planning session to decompose a goal."""
    console.print("[yellow]Plan command not yet implemented.[/yellow]")
    raise typer.Exit(code=1)


@app.command()
def run(
    project_root: Annotated[
        Path, typer.Option("--project-root", help="Project root directory")
    ] = Path("."),
) -> None:
    """Execute ready leaf nodes as parallel ralph loops."""
    console.print("[yellow]Run command not yet implemented.[/yellow]")
    raise typer.Exit(code=1)
