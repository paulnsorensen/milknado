"""MCP tools for the wiki <-> milknado roadmap crossover.

Coordinator-level tools alongside the other graph tools: import seeds the graph
from the wiki, export harvests execution state back into it.
"""

from __future__ import annotations

from pathlib import Path

from milknado.adapters.hallouminate import HallouminateIndexer
from milknado.domains.wiki import (
    export_roadmap,
    import_roadmap,
    resolve_roadmap_node,
    wiki_root,
)
from milknado.mcp._core import mcp, open_graph, resolve_project_root


def _wiki_root_for(project_root: str) -> tuple[Path, Path]:
    root = resolve_project_root(project_root or None)
    return root, wiki_root(root)


@mcp.tool()
def milknado_roadmap_import(roadmap_slug: str, project_root: str = "") -> dict:
    """Seed milknado roadmap + goal nodes from the wiki roadmap directory.

    Reads <project_root>/.hallouminate/wiki/roadmaps/<roadmap_slug>/; idempotent
    (re-import creates no duplicates). Seeds the goal layer only — task
    decomposition stays a separate milknado_plan_batches step.
    """
    root, wiki_root = _wiki_root_for(project_root)
    graph, _cfg = open_graph(root)
    try:
        result = import_roadmap(wiki_root, roadmap_slug, graph)
    finally:
        graph.close()
    return {
        "roadmap_node_id": result.roadmap_node_id,
        "goal_node_ids": result.goal_node_ids,
        "created": result.created_count,
        "reused": result.reused_count,
        "stamped": result.stamped,
    }


@mcp.tool()
def milknado_roadmap_export(roadmap_slug: str, project_root: str = "") -> dict:
    """Harvest milknado execution state back into the wiki goal files.

    Overwrites only each goal's harvest block + status/last_synced; human-authored
    Intent/Acceptance is left byte-identical. A goal node with no matching file
    gets a fresh goal file created for it.
    """
    root, wiki_root = _wiki_root_for(project_root)
    graph, _cfg = open_graph(root)
    indexer = HallouminateIndexer()
    try:
        node = resolve_roadmap_node(graph, wiki_root, roadmap_slug)
        result = export_roadmap(graph, node.id, wiki_root, indexer)
    finally:
        graph.close()
    return {
        "files_written": result.files_written,
        "files_created": result.files_created,
        "index_status": result.index.status,
        "index_detail": result.index.detail,
    }
