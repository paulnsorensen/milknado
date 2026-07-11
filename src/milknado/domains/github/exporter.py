"""Harvest milknado execution state onto a bound GitHub Project.

The milknado-owned half of the GitHub membrane: for each bound goal it writes the
Status single-select + harvest text field, and — only when the goal carries a
`wiki_ref` (wiki-origin projection) — overwrites the Issue body from the wiki
Intent. A github-origin goal's Issue body is never touched (that region is
human-owned). Field ownership, not byte ranges, is the membrane here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from milknado.adapters.gh import (
    gh_field_list,
    gh_issue_edit_body,
    gh_item_edit,
    gh_item_list,
    gh_preflight,
    gh_project_view,
)
from milknado.domains.common import MikadoNode, NodeKind, build_harvest_summary
from milknado.domains.github._fields import (
    HARVEST_FIELD_NAME,
    STATUS_FIELD_NAME,
    find_field,
    find_option_id,
    format_harvest_text,
    status_option_name,
)
from milknado.domains.github._intent import goal_file_map, goal_intent
from milknado.domains.graph import MikadoGraph


@dataclass(frozen=True)
class GithubExportResult:
    goals_exported: int
    bodies_overwritten: int


def export_github_roadmap(
    graph: MikadoGraph,
    roadmap_node_id: int,
    wiki_root: Path,
    *,
    owner: str,
    number: int,
) -> GithubExportResult:
    """Write Status + harvest for each bound goal; mirror the body for wiki-origin."""
    roadmap = graph.get_node(roadmap_node_id)
    if roadmap is None or roadmap.kind != NodeKind.ROADMAP:
        raise ValueError(f"node {roadmap_node_id} is not a roadmap node")
    if roadmap.github_ref is None:
        raise ValueError(
            f"roadmap node {roadmap_node_id} has no github_ref; bind or import it first"
        )
    gh_preflight()
    status_field, harvest_field = _resolve_fields(owner, number)
    url_by_item = {item["id"]: item.get("url") for item in gh_item_list(owner, number)}
    file_map = goal_file_map(wiki_root, roadmap.wiki_ref) if roadmap.wiki_ref else {}

    goals_exported = 0
    bodies_overwritten = 0
    for goal in graph.get_children(roadmap_node_id):
        if goal.kind != NodeKind.GOAL or goal.github_ref is None:
            continue
        _write_status(goal, roadmap.github_ref, status_field)
        _write_harvest(graph, goal, roadmap.github_ref, harvest_field)
        goals_exported += 1
        if goal.wiki_ref is not None and url_by_item.get(goal.github_ref):
            gh_issue_edit_body(url_by_item[goal.github_ref], goal_intent(goal, file_map))
            bodies_overwritten += 1
    return GithubExportResult(goals_exported=goals_exported, bodies_overwritten=bodies_overwritten)


def resolve_github_roadmap_node(graph: MikadoGraph, owner: str, number: int) -> MikadoNode:
    """Find the milknado roadmap node bound to a Project, via its PVT node id."""
    gh_preflight()
    project = gh_project_view(owner, number)
    node = graph.find_node_by_github_ref(project["id"])
    if node is None:
        raise LookupError(f"project {owner}/{number} not bound to milknado yet")
    return node


def _resolve_fields(owner: str, number: int) -> tuple[dict, dict]:
    fields = gh_field_list(owner, number)
    status_field = find_field(fields, STATUS_FIELD_NAME)
    harvest_field = find_field(fields, HARVEST_FIELD_NAME)
    if status_field is None or harvest_field is None:
        raise ValueError(
            f"project is missing the {STATUS_FIELD_NAME!r}/{HARVEST_FIELD_NAME!r} "
            "fields; run bind first"
        )
    return status_field, harvest_field


def _write_status(goal: MikadoNode, project_id: str, status_field: dict) -> None:
    option = status_option_name(goal.status.value)
    if option is None:
        return
    option_id = find_option_id(status_field, option)
    if option_id is not None:
        gh_item_edit(
            project_id, goal.github_ref, status_field["id"], single_select_option_id=option_id
        )


def _write_harvest(
    graph: MikadoGraph, goal: MikadoNode, project_id: str, harvest_field: dict
) -> None:
    text = format_harvest_text(build_harvest_summary(graph, goal))
    gh_item_edit(project_id, goal.github_ref, harvest_field["id"], text=text)
