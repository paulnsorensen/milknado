"""Tests for domains/execution/pr_stack.py."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from milknado.domains.execution.pr_stack import (
    _gh_create_pr,
    _topo_sort_group,
    build_overlap_groups,
    open_stacked_prs,
)


def _make_graph(ownership: dict[int, list[str]], parents: dict[int, int | None] | None = None):
    """Build a minimal duck-typed graph for pr_stack tests."""
    graph = MagicMock()
    graph.get_file_ownership.side_effect = lambda nid: ownership.get(nid, [])

    from milknado.domains.common.types import MikadoNode, NodeStatus

    def get_node(nid: int):
        parent_id = (parents or {}).get(nid)
        return MikadoNode(
            id=nid, description=f"task {nid}", parent_id=parent_id, status=NodeStatus.DONE
        )

    graph.get_node.side_effect = get_node
    return graph


class TestBuildOverlapGroups:
    def test_empty_returns_empty(self) -> None:
        graph = _make_graph({})
        assert build_overlap_groups([], graph) == []

    def test_single_node_is_own_group(self) -> None:
        graph = _make_graph({1: ["a.py"]})
        groups = build_overlap_groups([1], graph)
        assert groups == [[1]]

    def test_disjoint_nodes_are_separate_groups(self) -> None:
        graph = _make_graph({1: ["a.py"], 2: ["b.py"]})
        groups = build_overlap_groups([1, 2], graph)
        assert len(groups) == 2
        flat = sorted(n for g in groups for n in g)
        assert flat == [1, 2]

    def test_overlapping_nodes_merge_into_one_group(self) -> None:
        graph = _make_graph({1: ["a.py", "b.py"], 2: ["b.py", "c.py"]})
        groups = build_overlap_groups([1, 2], graph)
        assert len(groups) == 1
        assert sorted(groups[0]) == [1, 2]

    def test_transitive_overlap_merges_chain(self) -> None:
        # 1↔2 share x.py, 2↔3 share y.py → all three in one group
        graph = _make_graph({1: ["x.py"], 2: ["x.py", "y.py"], 3: ["y.py"]})
        groups = build_overlap_groups([1, 2, 3], graph)
        assert len(groups) == 1
        assert sorted(groups[0]) == [1, 2, 3]

    def test_no_file_ownership_makes_singletons(self) -> None:
        graph = _make_graph({1: [], 2: [], 3: []})
        groups = build_overlap_groups([1, 2, 3], graph)
        assert len(groups) == 3

    def test_partial_overlap_produces_two_groups(self) -> None:
        # 1↔2 overlap; 3 is disjoint
        graph = _make_graph({1: ["a.py"], 2: ["a.py"], 3: ["z.py"]})
        groups = build_overlap_groups([1, 2, 3], graph)
        assert len(groups) == 2
        sizes = sorted(len(g) for g in groups)
        assert sizes == [1, 2]


class TestTopoSortGroup:
    def test_single_node_returns_itself(self) -> None:
        graph = _make_graph({}, parents={1: None})
        assert _topo_sort_group([1], graph) == [1]

    def test_parent_child_orders_parent_first(self) -> None:
        # node 2 is a child of node 1 → 1 should come before 2
        graph = _make_graph({}, parents={1: None, 2: 1})
        result = _topo_sort_group([1, 2], graph)
        assert result.index(1) < result.index(2)

    def test_independent_nodes_sorted_by_id(self) -> None:
        graph = _make_graph({}, parents={3: None, 1: None, 2: None})
        result = _topo_sort_group([3, 1, 2], graph)
        assert result == [1, 2, 3]

    def test_chain_ordered_prerequisite_first(self) -> None:
        # 1 ← 2 ← 3 (3 depends on 2 depends on 1)
        graph = _make_graph({}, parents={1: None, 2: 1, 3: 2})
        result = _topo_sort_group([1, 2, 3], graph)
        assert result == [1, 2, 3]


class TestOpenStackedPrs:
    def test_empty_completed_branches_returns_empty(self) -> None:
        graph = _make_graph({})
        result = open_stacked_prs({}, graph, "main", Path("/repo"))
        assert result == []

    @patch("milknado.domains.execution.pr_stack._gh_create_pr")
    def test_singleton_targets_base_branch(self, mock_gh: MagicMock) -> None:
        mock_gh.return_value = "https://github.com/org/repo/pull/1"
        graph = _make_graph({1: ["a.py"]})
        result = open_stacked_prs({1: "milknado/1-foo"}, graph, "main", Path("/repo"))
        assert len(result) == 1
        assert result[0].base_branch == "main"
        assert result[0].branch == "milknado/1-foo"
        assert result[0].pr_url == "https://github.com/org/repo/pull/1"

    @patch("milknado.domains.execution.pr_stack._gh_create_pr")
    def test_two_overlapping_nodes_stack_second_onto_first(self, mock_gh: MagicMock) -> None:
        mock_gh.side_effect = [
            "https://github.com/org/repo/pull/1",
            "https://github.com/org/repo/pull/2",
        ]
        graph = _make_graph({1: ["a.py"], 2: ["a.py"]}, parents={1: None, 2: 1})
        result = open_stacked_prs(
            {1: "milknado/1-a", 2: "milknado/2-b"}, graph, "main", Path("/repo")
        )
        assert len(result) == 2
        first = next(pr for pr in result if pr.node_id == 1)
        second = next(pr for pr in result if pr.node_id == 2)
        assert first.base_branch == "main"
        assert second.base_branch == first.branch

    @patch("milknado.domains.execution.pr_stack._gh_create_pr")
    def test_disjoint_nodes_each_target_base_branch(self, mock_gh: MagicMock) -> None:
        mock_gh.side_effect = [
            "https://github.com/org/repo/pull/1",
            "https://github.com/org/repo/pull/2",
        ]
        graph = _make_graph({1: ["a.py"], 2: ["b.py"]})
        result = open_stacked_prs(
            {1: "milknado/1-a", 2: "milknado/2-b"}, graph, "main", Path("/repo")
        )
        assert len(result) == 2
        assert all(pr.base_branch == "main" for pr in result)


class TestGhCreatePr:
    @patch("milknado.domains.execution.pr_stack.subprocess.run")
    def test_returns_pr_url_from_stdout(self, mock_run: MagicMock) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            [], 0, "https://github.com/org/repo/pull/42\n", ""
        )
        url = _gh_create_pr("feat", "main", "My PR", "body", Path("/repo"))
        assert url == "https://github.com/org/repo/pull/42"

    @patch("milknado.domains.execution.pr_stack.subprocess.run")
    def test_calls_gh_pr_create_with_correct_args(self, mock_run: MagicMock) -> None:
        mock_run.return_value = subprocess.CompletedProcess([], 0, "https://x/1\n", "")
        _gh_create_pr("feat/branch", "main", "Title", "Body", Path("/repo"))
        args = mock_run.call_args[0][0]
        assert args[0] == "gh"
        assert "--head" in args and "feat/branch" in args
        assert "--base" in args and "main" in args
        assert "--title" in args and "Title" in args

    @patch("milknado.domains.execution.pr_stack.subprocess.run")
    def test_raises_on_nonzero_exit(self, mock_run: MagicMock) -> None:
        mock_run.side_effect = subprocess.CalledProcessError(1, "gh")
        with pytest.raises(subprocess.CalledProcessError):
            _gh_create_pr("feat", "main", "T", "B", Path("/repo"))
