from __future__ import annotations

import pytest

from milknado.domains.common import NodeStatus
from milknado.domains.graph import MikadoGraph, render_tree, summarize


@pytest.fixture()
def graph(tmp_path):
    g = MikadoGraph(tmp_path / "test.db")
    yield g
    g.close()


class TestSummarize:
    def test_empty_graph(self, graph):
        s = summarize(graph)
        assert s.total == 0
        assert s.done == 0
        assert s.pct_complete == 0.0
        assert s.ready == []
        assert s.conflicts == []

    def test_single_node(self, graph):
        graph.add_node("Root goal")
        s = summarize(graph)
        assert s.total == 1
        assert s.done == 0
        assert s.pct_complete == 0.0
        assert len(s.ready) == 1
        assert s.ready[0].description == "Root goal"

    def test_completion_percentage(self, graph):
        root = graph.add_node("Root")
        c1 = graph.add_node("Child 1", parent_id=root.id)
        graph.add_node("Child 2", parent_id=root.id)
        graph.mark_running(c1.id)
        graph.mark_done(c1.id)
        s = summarize(graph)
        assert s.total == 3
        assert s.done == 1
        assert s.pct_complete == pytest.approx(33.3, abs=0.1)

    def test_counts_by_status(self, graph):
        root = graph.add_node("Root")
        c1 = graph.add_node("Child 1", parent_id=root.id)
        c2 = graph.add_node("Child 2", parent_id=root.id)
        c3 = graph.add_node("Child 3", parent_id=root.id)
        graph.mark_running(c1.id)
        graph.mark_running(c2.id)
        graph.mark_failed(c2.id)
        graph.mark_blocked(c3.id)
        s = summarize(graph)
        assert s.running == 1
        assert s.failed == 1
        assert s.blocked == 1

    def test_ready_excludes_non_pending(self, graph):
        root = graph.add_node("Root")
        c1 = graph.add_node("Child 1", parent_id=root.id)
        graph.mark_running(c1.id)
        s = summarize(graph)
        assert all(n.status == NodeStatus.PENDING for n in s.ready)

    def test_conflicts_detected(self, graph):
        root = graph.add_node("Root")
        c1 = graph.add_node("Child 1", parent_id=root.id)
        c2 = graph.add_node("Child 2", parent_id=root.id)
        graph.set_file_ownership(c1.id, ["src/main.py"])
        graph.set_file_ownership(c2.id, ["src/main.py", "src/other.py"])
        s = summarize(graph)
        assert len(s.conflicts) == 1
        assert "src/main.py" in s.conflicts[0][2]

    def test_no_conflicts_when_disjoint(self, graph):
        root = graph.add_node("Root")
        c1 = graph.add_node("Child 1", parent_id=root.id)
        c2 = graph.add_node("Child 2", parent_id=root.id)
        graph.set_file_ownership(c1.id, ["src/a.py"])
        graph.set_file_ownership(c2.id, ["src/b.py"])
        s = summarize(graph)
        assert s.conflicts == []


class TestRenderTree:
    def test_empty_graph(self, graph):
        output = render_tree(graph)
        assert "No nodes" in output

    def test_single_root(self, graph):
        graph.add_node("My root goal")
        output = render_tree(graph)
        assert "My root goal" in output
        assert "0%" in output or "0/" in output

    def test_tree_structure(self, graph):
        root = graph.add_node("Root")
        graph.add_node("Child A", parent_id=root.id)
        graph.add_node("Child B", parent_id=root.id)
        output = render_tree(graph)
        assert "Root" in output
        assert "Child A" in output
        assert "Child B" in output

    def test_nested_tree(self, graph):
        root = graph.add_node("Root")
        mid = graph.add_node("Middle", parent_id=root.id)
        graph.add_node("Leaf", parent_id=mid.id)
        output = render_tree(graph)
        assert "Root" in output
        assert "Middle" in output
        assert "Leaf" in output

    def test_done_nodes_in_output(self, graph):
        root = graph.add_node("Root")
        c1 = graph.add_node("Done child", parent_id=root.id)
        graph.mark_running(c1.id)
        graph.mark_done(c1.id)
        output = render_tree(graph)
        assert "Done child" in output
        assert "1/2" in output

    def test_ready_nodes_listed(self, graph):
        root = graph.add_node("Root")
        graph.add_node("Ready leaf", parent_id=root.id)
        output = render_tree(graph)
        assert "Ready" in output
        assert "Ready leaf" in output

    def test_conflicts_displayed(self, graph):
        root = graph.add_node("Root")
        c1 = graph.add_node("A", parent_id=root.id)
        c2 = graph.add_node("B", parent_id=root.id)
        graph.set_file_ownership(c1.id, ["shared.py"])
        graph.set_file_ownership(c2.id, ["shared.py"])
        output = render_tree(graph)
        assert "Conflict" in output
        assert "shared.py" in output

    def test_running_shows_worktree(self, graph):
        root = graph.add_node("Root")
        c1 = graph.add_node("Worker", parent_id=root.id)
        graph.mark_running(c1.id)
        graph._conn.execute(
            "UPDATE nodes SET worktree_path = ? WHERE id = ?",
            ("/tmp/wt-1", c1.id),
        )
        graph._conn.commit()
        output = render_tree(graph)
        assert "/tmp/wt-1" in output
