"""Harvest milknado execution state onto a bound GitHub Project.

The milknado-owned half of the GitHub membrane: for each bound goal it writes the
Status single-select + harvest text field, and — only when the goal carries a
`wiki_ref` (wiki-origin projection) — overwrites the Issue body from the wiki
Intent. A github-origin goal's Issue body is never touched (that region is
human-owned). Field ownership, not byte ranges, is the membrane here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from milknado.domains.common import MikadoNode, NodeKind
from milknado.domains.github._fields import (
    HARVEST_FIELD_NAME,
    STATUS_FIELD_NAME,
    find_field,
    find_option_id,
    status_option_name,
)
from milknado.domains.github._intent import goal_file_map, goal_intent
from milknado.domains.github.ports import GithubProjectPort
from milknado.domains.graph import MikadoGraph
from milknado.domains.reporting import build_harvest_summary, format_harvest_text

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GithubExportResult:
    goals_exported: int
    bodies_overwritten: int


@dataclass(frozen=True)
class _ProjectFields:
    project_id: str
    status: dict
    harvest: dict


def export_github_roadmap(
    graph: MikadoGraph,
    roadmap_node_id: int,
    wiki_root: Path,
    github: GithubProjectPort,
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
    github.preflight()
    status_field, harvest_field = _resolve_fields(github, owner, number)
    fields = _ProjectFields(roadmap.github_ref, status_field, harvest_field)
    url_by_item = {
        item_id: item.get("url")
        for item in github.item_list(owner, number)
        if (item_id := item.get("id"))
    }
    file_map = goal_file_map(wiki_root, roadmap.wiki_ref) if roadmap.wiki_ref else {}

    goals_exported = 0
    bodies_overwritten = 0
    for goal in graph.get_children(roadmap_node_id):
        if goal.kind != NodeKind.GOAL or goal.github_ref is None:
            continue
        _write_status(github, goal, fields)
        _write_harvest(github, graph, goal, fields)
        goals_exported += 1
        if goal.wiki_ref is not None and url_by_item.get(goal.github_ref):
            github.issue_edit_body(
                url_by_item[goal.github_ref], goal_intent(goal, file_map, wiki_root)
            )
            bodies_overwritten += 1
    return GithubExportResult(goals_exported=goals_exported, bodies_overwritten=bodies_overwritten)


def resolve_github_roadmap_node(
    graph: MikadoGraph, owner: str, number: int, github: GithubProjectPort
) -> MikadoNode:
    """Find the milknado roadmap node bound to a Project, via its PVT node id."""
    github.preflight()
    project = github.project_view(owner, number)
    node = graph.find_node_by_github_ref(project["id"])
    if node is None:
        raise LookupError(f"project {owner}/{number} not bound to milknado yet")
    return node


def _resolve_fields(github: GithubProjectPort, owner: str, number: int) -> tuple[dict, dict]:
    fields = github.field_list(owner, number)
    status_field = find_field(fields, STATUS_FIELD_NAME)
    harvest_field = find_field(fields, HARVEST_FIELD_NAME)
    if status_field is None or harvest_field is None:
        raise ValueError(
            f"project is missing the {STATUS_FIELD_NAME!r}/{HARVEST_FIELD_NAME!r} "
            "fields; run bind first"
        )
    return status_field, harvest_field


def _write_status(github: GithubProjectPort, goal: MikadoNode, fields: _ProjectFields) -> None:
    option = status_option_name(goal.status.value)
    if option is None:
        return
    option_id = find_option_id(fields.status, option)
    if option_id is None:
        _logger.warning(
            "goal %s: status option %r not found on field %r; skipping status write",
            goal.id,
            option,
            fields.status.get("name"),
        )
        return
    github.item_edit(
        fields.project_id,
        goal.github_ref,
        fields.status["id"],
        single_select_option_id=option_id,
    )


def _write_harvest(
    github: GithubProjectPort,
    graph: MikadoGraph,
    goal: MikadoNode,
    fields: _ProjectFields,
) -> None:
    text = format_harvest_text(build_harvest_summary(graph, goal))
    github.item_edit(fields.project_id, goal.github_ref, fields.harvest["id"], text=text)
