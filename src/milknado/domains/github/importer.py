"""Import a GitHub Project v2 into milknado roadmap + goal nodes (github-origin).

Seeds the goal layer only: the Project becomes a roadmap node (github_ref = the
PVT project id) and each Project item becomes a goal node (github_ref = the PVTI
item id). Keyed by the opaque node ids so re-import is idempotent via
UNIQUE(github_ref); creates no task nodes.

Only the Issue title is read (into the node title). The spec's membrane table
lists "Issue body -> goal Intent", but the node model has no intent field — a
MikadoNode carries only `description`, and ADR-2 commits to a single additive
column (github_ref). Mirroring the body onto the node would need a new column the
spec's schema section never introduced, and nothing would ever read it: export
never writes a github-origin Issue body (it stays human-owned in GitHub). So the
body is deliberately left GitHub-owned and not stored (recorded as a decision in
the spec's membrane note).
"""

from __future__ import annotations

from dataclasses import dataclass

from milknado.adapters.gh import gh_item_list, gh_preflight, gh_project_view
from milknado.domains.common import NodeKind, NodeSpec
from milknado.domains.graph import MikadoGraph


@dataclass(frozen=True)
class GithubImportResult:
    roadmap_node_id: int
    goal_node_ids: dict[str, int]  # Project item id -> goal node id
    created_count: int
    reused_count: int


def import_github_roadmap(graph: MikadoGraph, owner: str, number: int) -> GithubImportResult:
    """Seed roadmap + goal nodes from a Project; idempotent, no task nodes."""
    gh_preflight()
    project = gh_project_view(owner, number)
    project_id = project["id"]
    created = 0
    reused = 0
    roadmap = graph.find_node_by_github_ref(project_id)
    if roadmap is None:
        roadmap = graph.add_node(
            project["title"], spec=NodeSpec(kind=NodeKind.ROADMAP, github_ref=project_id)
        )
        created += 1
    else:
        reused += 1
    goal_node_ids: dict[str, int] = {}
    for item in gh_item_list(owner, number):
        item_id = item.get("id")
        if not item_id:
            continue  # draft/malformed item with no node id — skip deterministically
        node = graph.find_node_by_github_ref(item_id)
        if node is None:
            node = graph.add_node(
                item.get("title") or "(untitled item)",
                parent_id=roadmap.id,
                spec=NodeSpec(kind=NodeKind.GOAL, github_ref=item_id),
            )
            created += 1
        else:
            reused += 1
        goal_node_ids[item_id] = node.id
    return GithubImportResult(
        roadmap_node_id=roadmap.id,
        goal_node_ids=goal_node_ids,
        created_count=created,
        reused_count=reused,
    )
