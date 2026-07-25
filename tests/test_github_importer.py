"""`import_github_roadmap`: github-origin seeding through an injected port."""

from __future__ import annotations

import pytest

from milknado.domains.common import NodeKind
from milknado.domains.github import GithubItem, GithubProject
from milknado.domains.github.importer import import_github_roadmap
from milknado.domains.graph import MikadoGraph

PROJECT = GithubProject(id="PVT_1", title="Peer Roadmap")
ITEMS = [
    GithubItem(id="PVTI_a", title="Goal A"),
    GithubItem(id="PVTI_b", title="Goal B"),
]


class FakeGithub:
    def __init__(self) -> None:
        self.items = list(ITEMS)

    def preflight(self) -> None:
        pass

    def project_view(self, _owner: str, _number: int) -> GithubProject:
        return PROJECT

    def item_list(self, _owner: str, _number: int) -> list[GithubItem]:
        return list(self.items)


@pytest.fixture
def github() -> FakeGithub:
    return FakeGithub()


def test_import_seeds_roadmap_and_goals(graph: MikadoGraph, github: FakeGithub) -> None:
    result = import_github_roadmap(graph, "acme", 1, github)
    roadmap = graph.get_node(result.roadmap_node_id)
    assert roadmap is not None
    assert roadmap.kind == NodeKind.ROADMAP
    assert roadmap.github_ref == "PVT_1"
    assert result.created_count == 3
    assert set(result.goal_node_ids) == {"PVTI_a", "PVTI_b"}


def test_goals_keyed_by_item_id_and_linked_to_roadmap(
    graph: MikadoGraph, github: FakeGithub
) -> None:
    result = import_github_roadmap(graph, "acme", 1, github)
    children = graph.get_children(result.roadmap_node_id)
    assert {c.github_ref for c in children} == {"PVTI_a", "PVTI_b"}
    assert all(c.kind == NodeKind.GOAL for c in children)
    assert graph.get_node(result.goal_node_ids["PVTI_a"]).github_ref == "PVTI_a"


def test_reimport_is_idempotent(graph: MikadoGraph, github: FakeGithub) -> None:
    first = import_github_roadmap(graph, "acme", 1, github)
    second = import_github_roadmap(graph, "acme", 1, github)
    assert second.created_count == 0
    assert second.reused_count == 3
    assert second.roadmap_node_id == first.roadmap_node_id
    assert len(graph.get_all_nodes()) == 3


def test_reimport_adopts_new_item_without_duplicating(
    graph: MikadoGraph, github: FakeGithub
) -> None:
    import_github_roadmap(graph, "acme", 1, github)
    github.items.append(GithubItem(id="PVTI_c", title="Goal C"))

    second = import_github_roadmap(graph, "acme", 1, github)

    assert second.created_count == 1
    assert second.reused_count == 3
    assert set(second.goal_node_ids) == {"PVTI_a", "PVTI_b", "PVTI_c"}
    assert len(graph.get_all_nodes()) == 4


def test_import_creates_no_task_nodes(graph: MikadoGraph, github: FakeGithub) -> None:
    import_github_roadmap(graph, "acme", 1, github)
    assert not any(n.kind == NodeKind.TASK for n in graph.get_all_nodes())
