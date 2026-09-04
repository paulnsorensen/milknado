"""Roadmap graph CLI commands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Literal

import typer

from milknado.adapters.hallouminate import HallouminateIndexer
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
from milknado.domains.wiki import (
    RoadmapModel,
    export_roadmap,
    import_roadmap,
    load_roadmap,
    render_html,
    render_mermaid,
    resolve_roadmap_dir,
    resolve_roadmap_node,
    roadmap_json,
    roadmap_schema,
    wiki_root,
)

roadmap_app = typer.Typer(name="roadmap", help="Import/export and inspect roadmap graphs")

_ProjectRoot = Annotated[Path, typer_option("--project-root", help="Project root directory")]
_Out = Annotated[Path | None, typer_option("--out", help="Write output to this path")]
_Format = Annotated[Literal["mermaid", "html"], typer_option("--format", help="Render format")]


def _roadmap_model(project_root: Path, slug: str) -> RoadmapModel:
    root = wiki_root(project_root)
    return load_roadmap(resolve_roadmap_dir(root, slug))


@roadmap_app.command("schema")
def roadmap_schema_cmd(out: _Out = None) -> None:
    """Print or write the canonical roadmap model JSON Schema."""
    try:
        _emit(json.dumps(roadmap_schema(), indent=2, sort_keys=True), out)
    except OSError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None


@roadmap_app.command("json")
def roadmap_json_cmd(
    slug: Annotated[str, typer_argument(help="Roadmap slug")],
    project_root: _ProjectRoot = DEFAULT_PROJECT_ROOT,
    out: _Out = None,
) -> None:
    """Print or write a canonical roadmap JSON instance."""
    try:
        payload = roadmap_json(_roadmap_model(project_root.resolve(), slug))
        _emit(json.dumps(payload, indent=2, sort_keys=True), out)
    except (OSError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None


@roadmap_app.command("render")
def roadmap_render_cmd(
    slug: Annotated[str, typer_argument(help="Roadmap slug")],
    format: _Format = "mermaid",
    project_root: _ProjectRoot = DEFAULT_PROJECT_ROOT,
    out: _Out = None,
) -> None:
    """Render a roadmap graph as Mermaid or self-contained HTML."""
    try:
        model = _roadmap_model(project_root.resolve(), slug)
        rendered = render_mermaid(model) if format == "mermaid" else render_html(model)
        _emit(rendered, out)
    except (OSError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None


@roadmap_app.command("import")
def import_roadmap_cmd(
    slug: Annotated[str, typer_argument(help="Roadmap slug (directory under roadmaps/)")],
    project_root: _ProjectRoot = DEFAULT_PROJECT_ROOT,
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
        + f"{result.reused_count} reused, {len(result.goal_node_ids)} goals."
    )


@roadmap_app.command("export")
def export_roadmap_cmd(
    slug: Annotated[str, typer_argument(help="Roadmap slug (directory under roadmaps/)")],
    project_root: _ProjectRoot = DEFAULT_PROJECT_ROOT,
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
        + f"{result.files_created} created, index={result.index.status}{detail}."
    )
