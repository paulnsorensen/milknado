"""`import_github_roadmap`: github-origin seeding through an injected port."""

from __future__ import annotations

from collections.abc import Sequence
from typing import final

import pytest

from milknado.domains.common import NodeKind
from milknado.domains.github import GithubField, GithubItem, GithubProject
from milknado.domains.github.importer import import_github_roadmap
from milknado.domains.graph import MikadoGraph

PROJECT = GithubProject(id="PVT_1", title="Peer Roadmap")
ITEMS = [
    GithubItem(id="PVTI_a", title="Goal A"),
    GithubItem(id="PVTI_b", title="Goal B"),
]


@final
class FakeGithub:
    def __init__(self) -> None:
        self.items: list[GithubItem] = list(ITEMS)

    def preflight(self) -> None:
        pass

    def project_view(self, owner: str, number: int) -> GithubProject:
        _ = (owner, number)
        return PROJECT

    def item_list(self, owner: str, number: int) -> list[GithubItem]:
        _ = (owner, number)
        return list(self.items)

    def item_add(self, owner: str, number: int, issue_url: str) -> str:
        _ = (owner, number, issue_url)
        raise AssertionError("item_add is not used by importer tests")

    def item_edit(  # noqa: PLR0913
        self,
        project_id: str,
        item_id: str,
        field_id: str,
        *,
        single_select_option_id: str | None = None,
        text: str | None = None,
    ) -> None:
        _ = (project_id, item_id, field_id, single_select_option_id, text)
        raise AssertionError("item_edit is not used by importer tests")

    def field_list(self, owner: str, number: int) -> list[GithubField]:
        _ = (owner, number)
        raise AssertionError("field_list is not used by importer tests")

    def field_create(self, number: int, owner: str, name: str, options: Sequence[str]) -> str:
        _ = (number, owner, name, options)
        raise AssertionError("field_create is not used by importer tests")

    def field_create_text(self, number: int, owner: str, name: str) -> str:
        _ = (number, owner, name)
        raise AssertionError("field_create_text is not used by importer tests")

    def issue_create(self, owner: str, repo: str, title: str, body: str) -> str:
        _ = (owner, repo, title, body)
        raise AssertionError("issue_create is not used by importer tests")

    def issue_edit_body(self, issue_url: str, body: str) -> None:
        _ = (issue_url, body)
        raise AssertionError("issue_edit_body is not used by importer tests")


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
    goal = graph.get_node(result.goal_node_ids["PVTI_a"])
    assert goal is not None
    assert goal.github_ref == "PVTI_a"


def test_reimport_is_idempotent(graph: MikadoGraph, github: FakeGithub) -> None:
    first = import_github_roadmap(graph, "acme", 1, github)
    second = import_github_roadmap(graph, "acme", 1, github)
    assert second.created_count == 0
    assert second.reused_count == 3
    assert second.roadmap_node_id == first.roadmap_node_id
    roadmap = graph.get_node(first.roadmap_node_id)
    assert roadmap is not None
    assert roadmap.github_ref == "PVT_1"
    assert len(graph.get_all_nodes()) == 3


def test_reimport_adopts_new_item_without_duplicating(
    graph: MikadoGraph, github: FakeGithub
) -> None:
    _ = import_github_roadmap(graph, "acme", 1, github)
    github.items.append(GithubItem(id="PVTI_c", title="Goal C"))

    second = import_github_roadmap(graph, "acme", 1, github)

    assert second.created_count == 1
    assert second.reused_count == 3
    assert set(second.goal_node_ids) == {"PVTI_a", "PVTI_b", "PVTI_c"}
    assert len(graph.get_all_nodes()) == 4


def test_import_creates_no_task_nodes(graph: MikadoGraph, github: FakeGithub) -> None:
    _ = import_github_roadmap(graph, "acme", 1, github)
    assert not any(n.kind == NodeKind.TASK for n in graph.get_all_nodes())
