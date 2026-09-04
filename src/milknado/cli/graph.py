"""Emergency graph-ops CLI — direct edge mutation outside the plan/roadmap flow."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import typer

from milknado.cli._helpers import (
    DEFAULT_PROJECT_ROOT,
    console,
    typer_argument,
    typer_option,
)
from milknado.cli._helpers import (
    emit as _emit,
)
from milknado.cli._helpers import (
    ensure_db as _ensure_db,
)
from milknado.cli._helpers import (
    load_or_default as _load_or_default,
)
from milknado.domains.graph import render_dot

edge_app = typer.Typer(name="edge", help="Direct edge operations on the Mikado graph")
graph_app = typer.Typer(name="graph", help="Archive lifecycle operations on the Mikado graph")


@graph_app.command("archive")
def archive(
    node_id: Annotated[int, typer_argument(help="Subtree root node ID")],
    project_root: Annotated[
        Path, typer_option("--project-root", help="Project root directory")
    ] = DEFAULT_PROJECT_ROOT,
) -> None:
    """Soft-hide an all-DONE subtree (reversible; refuses when any node is live)."""
    project_root = project_root.resolve()
    config, plugins = _load_or_default(project_root)
    graph = _ensure_db(config, plugins)

    try:
        try:
            archived = graph.archive_subtree(node_id)
        except ValueError as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(code=1) from None
        console.print(f"Archived {archived} node(s) under {node_id}")
    finally:
        graph.close()


@graph_app.command("unarchive")
def unarchive(
    node_id: Annotated[int, typer_argument(help="Subtree root node ID")],
    project_root: Annotated[
        Path, typer_option("--project-root", help="Project root directory")
    ] = DEFAULT_PROJECT_ROOT,
) -> None:
    """Restore an archived subtree; refuses while an ancestor stays archived."""
    project_root = project_root.resolve()
    config, plugins = _load_or_default(project_root)
    graph = _ensure_db(config, plugins)

    try:
        try:
            unarchived = graph.unarchive_subtree(node_id)
        except ValueError as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(code=1) from None
        console.print(f"Unarchived {unarchived} node(s) under {node_id}")
    finally:
        graph.close()


@edge_app.command("add")
def add(
    parent_id: Annotated[int, typer_argument(help="Parent node ID")],
    child_id: Annotated[int, typer_argument(help="Child node ID")],
    project_root: Annotated[
        Path, typer_option("--project-root", help="Project root directory")
    ] = DEFAULT_PROJECT_ROOT,
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
            _ = graph.add_edge(parent_id, child_id)
        except ValueError as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(code=1) from None

        console.print(f"Added edge {parent_id} -> {child_id}")
    finally:
        graph.close()


@graph_app.command("export")
def export(
    format: Annotated[Literal["dot"], typer_option("--format", help="Export format")] = "dot",
    out: Annotated[Path | None, typer_option("--out", help="Write output to this path")] = None,
    include_archived: Annotated[
        bool, typer_option("--include-archived", help="Include archived nodes")
    ] = False,
    project_root: Annotated[
        Path, typer_option("--project-root", help="Project root directory")
    ] = DEFAULT_PROJECT_ROOT,
) -> None:
    _ = format
    project_root = project_root.resolve()
    config, plugins = _load_or_default(project_root)
    graph = _ensure_db(config, plugins)
    try:
        nodes = graph.get_all_nodes(include_archived=include_archived)
        children = graph.get_children_map(include_archived=include_archived)
        content = render_dot(nodes, children)
    finally:
        graph.close()

    try:
        _emit(content, out)
    except OSError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None
