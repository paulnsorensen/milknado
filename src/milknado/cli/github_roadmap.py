"""`milknado github-roadmap import|bind|export` — CLI veneer over the github slice."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from milknado.adapters.gh import GhProjectAdapter, GhTransportError
from milknado.cli._helpers import _ensure_db, _load_or_default, console
from milknado.domains.github import (
    bind_github_project,
    export_github_roadmap,
    import_github_roadmap,
    resolve_github_roadmap_node,
)
from milknado.domains.wiki import resolve_roadmap_node, wiki_root

github_roadmap_app = typer.Typer(
    name="github-roadmap", help="Import/bind/export roadmaps to a GitHub Project v2"
)

_ProjectRoot = Annotated[Path, typer.Option("--project-root", help="Project root directory")]
_Owner = Annotated[str, typer.Argument(help="Project owner login (user/org)")]
_Number = Annotated[int, typer.Argument(help="Project number")]

_ERRORS = (FileNotFoundError, ValueError, LookupError, GhTransportError)


@github_roadmap_app.command("import")
def import_cmd(owner: _Owner, number: _Number, project_root: _ProjectRoot = Path(".")) -> None:
    """Seed milknado roadmap + goal nodes from a GitHub Project (github-origin)."""
    project_root = project_root.resolve()
    github = GhProjectAdapter()
    config, plugins = _load_or_default(project_root)
    graph = _ensure_db(config, plugins)
    try:
        result = import_github_roadmap(graph, owner, number, github)
    except _ERRORS as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None
    finally:
        graph.close()
    console.print(
        f"Imported project {owner}/{number}: {result.created_count} created, "
        f"{result.reused_count} reused, {len(result.goal_node_ids)} goals."
    )


@github_roadmap_app.command("bind")
def bind_cmd(
    slug: Annotated[str, typer.Argument(help="Wiki roadmap slug to project")],
    project_root: _ProjectRoot = Path("."),
) -> None:
    """Project a wiki-origin roadmap onto a GitHub Project (one-time bind)."""
    project_root = project_root.resolve()
    roadmap_root = wiki_root(project_root)
    config, plugins = _load_or_default(project_root)
    github = GhProjectAdapter()
    graph = _ensure_db(config, plugins)
    try:
        node = resolve_roadmap_node(graph, roadmap_root, slug)
        result = bind_github_project(graph, node.id, roadmap_root, github)
    except _ERRORS as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None
    finally:
        graph.close()
    console.print(
        f"Bound roadmap {slug}: {result.issues_created} issues, "
        f"{result.items_added} items, field_created={result.field_created}."
    )


@github_roadmap_app.command("export")
def export_cmd(owner: _Owner, number: _Number, project_root: _ProjectRoot = Path(".")) -> None:
    """Harvest milknado execution state onto the bound Project item fields."""
    project_root = project_root.resolve()
    roadmap_root = wiki_root(project_root)
    config, plugins = _load_or_default(project_root)
    github = GhProjectAdapter()
    graph = _ensure_db(config, plugins)
    try:
        node = resolve_github_roadmap_node(graph, owner, number, github)
        result = export_github_roadmap(
            graph, node.id, roadmap_root, github, owner=owner, number=number
        )
    except _ERRORS as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None
    finally:
        graph.close()
    console.print(
        f"Exported project {owner}/{number}: {result.goals_exported} goals, "
        f"{result.bodies_overwritten} bodies overwritten."
    )
