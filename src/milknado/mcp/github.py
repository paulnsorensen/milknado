"""MCP tools for the GitHub Projects v2 <-> milknado roadmap crossover.

Thin veneers over the tested github slice, mirroring mcp_wiki: import seeds the
graph from a Project, bind projects a wiki-origin roadmap, export harvests state
onto the Project item fields.
"""

from __future__ import annotations

from pathlib import Path

from milknado.adapters.gh import GhProjectAdapter
from milknado.domains.github import (
    bind_github_project,
    export_github_roadmap,
    import_github_roadmap,
    resolve_github_roadmap_node,
)
from milknado.domains.wiki import resolve_roadmap_node, wiki_root
from milknado.mcp._core import mcp, open_graph, resolve_project_root


def _roots(project_root: str) -> tuple[Path, Path]:
    root = resolve_project_root(project_root or None)
    return root, wiki_root(root)


@mcp.tool()
def milknado_github_roadmap_import(owner: str, number: int, project_root: str = "") -> dict:
    """Seed milknado roadmap + goal nodes from a GitHub Project (github-origin).

    Idempotent via UNIQUE(github_ref); seeds the goal layer only (no task nodes).
    """
    root, _wiki = _roots(project_root)
    graph, _cfg = open_graph(root)
    github = GhProjectAdapter()
    try:
        result = import_github_roadmap(graph, owner, number, github)
    finally:
        graph.close()
    return {
        "roadmap_node_id": result.roadmap_node_id,
        "goal_node_ids": result.goal_node_ids,
        "created": result.created_count,
        "reused": result.reused_count,
    }


@mcp.tool()
def milknado_github_roadmap_bind(roadmap_slug: str, project_root: str = "") -> dict:
    """Project a wiki-origin roadmap onto a GitHub Project (one-time bind).

    Resolves the binding from the roadmap index.md `github_project`/`github_repo`
    frontmatter, creates one Issue per goal, adds items, and bootstraps the
    milknado-owned Status + harvest fields create-once.
    """
    root, wiki_root = _roots(project_root)
    graph, _cfg = open_graph(root)
    github = GhProjectAdapter()
    try:
        node = resolve_roadmap_node(graph, wiki_root, roadmap_slug)
        result = bind_github_project(graph, node.id, wiki_root, github)
    finally:
        graph.close()
    return {
        "roadmap_node_id": result.roadmap_node_id,
        "issues_created": result.issues_created,
        "items_added": result.items_added,
        "field_created": result.field_created,
    }


@mcp.tool()
def milknado_github_roadmap_export(owner: str, number: int, project_root: str = "") -> dict:
    """Harvest milknado execution state onto the bound Project item fields.

    Writes Status + harvest per bound goal; overwrites the Issue body only for
    wiki-origin goals (never for github-origin).
    """
    root, wiki_root = _roots(project_root)
    graph, _cfg = open_graph(root)
    github = GhProjectAdapter()
    try:
        node = resolve_github_roadmap_node(graph, owner, number, github)
        result = export_github_roadmap(
            graph, node.id, wiki_root, github, owner=owner, number=number
        )
    finally:
        graph.close()
    return {
        "goals_exported": result.goals_exported,
        "bodies_overwritten": result.bodies_overwritten,
    }
