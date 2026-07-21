"""`milknado roadmap import|export` — thin CLI veneer over the wiki crossover."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from milknado.adapters.hallouminate import HallouminateIndexer
from milknado.cli._helpers import _ensure_db, _load_or_default, console
from milknado.domains.wiki import (
    export_roadmap,
    import_roadmap,
    resolve_roadmap_node,
    wiki_root,
)

roadmap_app = typer.Typer(name="roadmap", help="Import/export roadmaps to the hallouminate wiki")

_ProjectRoot = Annotated[Path, typer.Option("--project-root", help="Project root directory")]


@roadmap_app.command("import")
def import_roadmap_cmd(
    slug: Annotated[str, typer.Argument(help="Roadmap slug (directory under roadmaps/)")],
    project_root: _ProjectRoot = Path("."),
) -> None:
    """Seed milknado roadmap + goal nodes from a wiki roadmap directory."""
    project_root = project_root.resolve()
    config, plugins = _load_or_default(project_root)
    graph = _ensure_db(config, plugins)
    try:
        result = import_roadmap(wiki_root(project_root), slug, graph)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None
    finally:
        graph.close()
    console.print(
        f"Imported roadmap {slug}: {result.created_count} created, "
        f"{result.reused_count} reused, {len(result.goal_node_ids)} goals."
    )


@roadmap_app.command("export")
def export_roadmap_cmd(
    slug: Annotated[str, typer.Argument(help="Roadmap slug (directory under roadmaps/)")],
    project_root: _ProjectRoot = Path("."),
) -> None:
    """Harvest milknado execution state back into the wiki goal files."""
    project_root = project_root.resolve()
    root = wiki_root(project_root)
    config, plugins = _load_or_default(project_root)
    graph = _ensure_db(config, plugins)
    indexer = HallouminateIndexer()
    try:
        node = resolve_roadmap_node(graph, root, slug)
        result = export_roadmap(graph, node.id, root, indexer)
    except (FileNotFoundError, ValueError, LookupError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None
    finally:
        graph.close()
    detail = f": {result.index.detail}" if result.index.detail else ""
    console.print(
        f"Exported roadmap {slug}: {result.files_written} updated, "
        f"{result.files_created} created, index={result.index.status}{detail}."
    )
