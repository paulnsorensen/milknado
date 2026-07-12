"""`import_github_roadmap`: github-origin seeding (Acceptance 1).

Seeds one roadmap node (github_ref = project id) + one goal node per Project item
(github_ref = item id), keyed by opaque node id so re-import is idempotent; never
creates task nodes. gh transport is monkeypatched — the importer's own `gh_*`
names are replaced so nothing shells out.
"""

from __future__ import annotations

import pytest

from milknado.domains.common import NodeKind
from milknado.domains.github import importer
from milknado.domains.github.importer import import_github_roadmap
from milknado.domains.graph import MikadoGraph

PROJECT = {"id": "PVT_1", "title": "Peer Roadmap"}
ITEMS = [
    {"id": "PVTI_a", "title": "Goal A"},
    {"id": "PVTI_b", "title": "Goal B"},
]


@pytest.fixture(autouse=True)
def _stub_gh(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(importer, "gh_preflight", lambda: None)
    monkeypatch.setattr(importer, "gh_project_view", lambda _o, _n: PROJECT)
    monkeypatch.setattr(importer, "gh_item_list", lambda _o, _n: list(ITEMS))


def test_import_seeds_roadmap_and_goals(graph: MikadoGraph) -> None:
    result = import_github_roadmap(graph, "acme", 1)
    roadmap = graph.get_node(result.roadmap_node_id)
    assert roadmap is not None
    assert roadmap.kind == NodeKind.ROADMAP
    assert roadmap.github_ref == "PVT_1"
    assert result.created_count == 3  # 1 roadmap + 2 goals
    assert set(result.goal_node_ids) == {"PVTI_a", "PVTI_b"}


def test_goals_keyed_by_item_id_and_linked_to_roadmap(graph: MikadoGraph) -> None:
    result = import_github_roadmap(graph, "acme", 1)
    children = graph.get_children(result.roadmap_node_id)
    assert {c.github_ref for c in children} == {"PVTI_a", "PVTI_b"}
    assert all(c.kind == NodeKind.GOAL for c in children)
    assert graph.get_node(result.goal_node_ids["PVTI_a"]).github_ref == "PVTI_a"


def test_reimport_is_idempotent(graph: MikadoGraph) -> None:
    first = import_github_roadmap(graph, "acme", 1)
    second = import_github_roadmap(graph, "acme", 1)
    assert second.created_count == 0
    assert second.reused_count == 3
    assert second.roadmap_node_id == first.roadmap_node_id
    assert len(graph.get_all_nodes()) == 3  # no duplicates


def test_reimport_adopts_new_item_without_duplicating(
    graph: MikadoGraph, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Acceptance 1 idempotency under change: a second import where the Project
    # gained an item creates ONLY the new goal, reuses the roadmap + prior goals,
    # and never duplicates a node keyed by an already-seen item id.
    import_github_roadmap(graph, "acme", 1)
    monkeypatch.setattr(
        importer,
        "gh_item_list",
        lambda _o, _n: [*ITEMS, {"id": "PVTI_c", "title": "Goal C"}],
    )
    second = import_github_roadmap(graph, "acme", 1)
    assert second.created_count == 1  # only PVTI_c
    assert second.reused_count == 3  # roadmap + PVTI_a + PVTI_b
    assert set(second.goal_node_ids) == {"PVTI_a", "PVTI_b", "PVTI_c"}
    assert len(graph.get_all_nodes()) == 4  # 1 roadmap + 3 goals, no dupes


def test_import_creates_no_task_nodes(graph: MikadoGraph) -> None:
    import_github_roadmap(graph, "acme", 1)
    assert not any(n.kind == NodeKind.TASK for n in graph.get_all_nodes())
