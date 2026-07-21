"""Emergency graph-ops CLI — direct edge mutation outside the plan/roadmap flow."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from milknado.cli._helpers import _ensure_db, _load_or_default, console

edge_app = typer.Typer(name="edge", help="Direct edge operations on the Mikado graph")


@edge_app.command("add")
def add(
    parent_id: Annotated[int, typer.Argument(help="Parent node ID")],
    child_id: Annotated[int, typer.Argument(help="Child node ID")],
    project_root: Annotated[
        Path, typer.Option("--project-root", help="Project root directory")
    ] = Path("."),
) -> None:
    """Add a prereq edge between two existing nodes, rejecting cycles."""
    project_root = project_root.resolve()
    config, plugins = _load_or_default(project_root)
    graph = _ensure_db(config, plugins)

    try:
        for node_id in (parent_id, child_id):
            if graph.get_node(node_id) is None:
                console.print(f"[red]Node {node_id} does not exist.[/red]")
                raise typer.Exit(code=1)

        try:
            graph.add_edge(parent_id, child_id)
        except ValueError as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(code=1) from None

        console.print(f"Added edge {parent_id} -> {child_id}")
    finally:
        graph.close()
