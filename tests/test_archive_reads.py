"""Archive-aware read projections (spec archive-rebalance, acceptance #8-#10).

#8: default forest walks (tree/summary/next/ready) exclude archived nodes.
#9: ``include_archived=True`` restores them; harvest keeps seeing archived
    DONE goals.
#10: display renders archived nodes with an ``[archived]`` marker when
    ``include_archived=True``.

Read-projection tests seed ``archived_at`` directly so failures stay isolated
from archive-mutation behavior.
"""

from __future__ import annotations

from datetime import UTC, datetime

from milknado.domains.common import NodeKind, NodeSpec, NodeStatus
from milknado.domains.graph import _reads, render_tree, summarize
from milknado.domains.graph.display import format_node
from milknado.domains.reporting.harvest import build_harvest_summary


def _archive(graph, *node_ids: int) -> None:
    now = datetime.now(UTC).isoformat()
    for node_id in node_ids:
        graph._conn.execute("UPDATE nodes SET archived_at = ? WHERE id = ?", (now, node_id))
    graph._conn.commit()


def _done_graph(graph):
    """Root goal with two done tasks; returns (root, task1, task2)."""
    root = graph.add_node("Root goal", spec=NodeSpec(kind=NodeKind.GOAL))
    t1 = graph.add_node("Task 1", parent_id=root.id)
    t2 = graph.add_node("Task 2", parent_id=root.id)
    for task in (t1, t2):
        graph.mark_running(task.id)
        graph.mark_done(task.id)
    return root, t1, t2


class TestDefaultReadsExcludeArchived:
    """Acceptance #8: without include_archived, archived nodes vanish."""

    def test_get_all_nodes_hides_archived(self, graph):
        root, t1, t2 = _done_graph(graph)
        _archive(graph, root.id, t1.id, t2.id)
        assert graph.get_all_nodes() == []

    def test_get_roots_hides_archived_root(self, graph):
        root, t1, t2 = _done_graph(graph)
        live = graph.add_node("Live root")
        _archive(graph, root.id, t1.id, t2.id)
        assert [n.id for n in graph.get_roots()] == [live.id]

    def test_get_root_skips_archived_first_root(self, graph):
        root, t1, t2 = _done_graph(graph)
        live = graph.add_node("Live root")
        _archive(graph, root.id, t1.id, t2.id)
        assert graph.get_root().id == live.id

    def test_get_children_hides_archived_children(self, graph):
        root, t1, t2 = _done_graph(graph)
        _archive(graph, t1.id, t2.id)
        assert graph.get_children(root.id) == []

    def test_children_map_hides_archived_subtree(self, graph):
        root, t1, t2 = _done_graph(graph)
        _archive(graph, root.id, t1.id, t2.id)
        assert graph.get_children_map() == {}

    def test_children_map_prunes_descendants_of_archived_node(self, graph):
        # Defensive: even if the archive cascade were violated and a child
        # stayed un-archived, an archived parent must hide its whole subtree.
        root = graph.add_node("Root")
        mid = graph.add_node("Middle", parent_id=root.id)
        leaf = graph.add_node("Leaf", parent_id=mid.id)
        _archive(graph, mid.id)
        children_map = graph.get_children_map()
        assert children_map.get(root.id, []) == []
        assert mid.id not in children_map
        assert all(n.id != leaf.id for kids in children_map.values() for n in kids)

    def test_get_leaves_hides_archived(self, graph):
        root, t1, t2 = _done_graph(graph)
        _archive(graph, t1.id)
        assert [n.id for n in graph.get_leaves()] == [t2.id]

    def test_ready_nodes_never_return_archived(self, graph):
        # todo_next backing read: an archived node must never come back,
        # even in the (invariant-violating) case of an archived PENDING node.
        root = graph.add_node("Root")
        live = graph.add_node("Live leaf", parent_id=root.id)
        shelved = graph.add_node("Shelved leaf", parent_id=root.id)
        _archive(graph, shelved.id)
        assert [n.id for n in graph.get_ready_nodes()] == [live.id]

    def test_node_summaries_hide_archived(self, graph):
        root, t1, t2 = _done_graph(graph)
        _archive(graph, root.id, t1.id, t2.id)
        assert graph.get_node_summaries() == []

    def test_get_node_point_lookup_still_returns_archived(self, graph):
        # Spec 5.1: get_node is a point lookup, not a forest projection —
        # callers addressing a node by id get the real row by default.
        root, t1, _ = _done_graph(graph)
        _archive(graph, t1.id)
        node = graph.get_node(t1.id)
        assert node is not None
        assert node.archived_at is not None
        direct = _reads.get_node(graph._conn, t1.id)
        assert direct is not None and direct.archived_at is not None

    def test_render_tree_hides_archived_by_default(self, graph):
        root, t1, t2 = _done_graph(graph)
        _archive(graph, t1.id, t2.id)
        output = render_tree(graph)
        assert "Root goal" in output
        assert "Task 1" not in output
        assert "Task 2" not in output
        assert "0/1" in output

    def test_summarize_excludes_archived_by_default(self, graph):
        root, t1, t2 = _done_graph(graph)
        _archive(graph, root.id, t1.id, t2.id)
        s = summarize(graph)
        assert s.total == 0
        assert s.done == 0


class TestIncludeArchivedRestores:
    """Acceptance #9: include_archived=True restores the full projection."""

    def test_get_all_nodes_includes_archived(self, graph):
        root, t1, t2 = _done_graph(graph)
        _archive(graph, root.id, t1.id, t2.id)
        nodes = _reads.get_all_nodes(graph._conn, include_archived=True)
        assert {n.id for n in nodes} == {root.id, t1.id, t2.id}

    def test_get_roots_includes_archived(self, graph):
        root, t1, t2 = _done_graph(graph)
        _archive(graph, root.id, t1.id, t2.id)
        roots = _reads.get_roots(graph._conn, include_archived=True)
        assert [n.id for n in roots] == [root.id]

    def test_get_children_includes_archived(self, graph):
        root, t1, t2 = _done_graph(graph)
        _archive(graph, t1.id, t2.id)
        children = _reads.get_children(graph._conn, root.id, include_archived=True)
        assert {n.id for n in children} == {t1.id, t2.id}

    def test_children_map_includes_archived(self, graph):
        root, t1, t2 = _done_graph(graph)
        _archive(graph, root.id, t1.id, t2.id)
        children_map = _reads.get_children_map(graph._conn, include_archived=True)
        assert {n.id for n in children_map[root.id]} == {t1.id, t2.id}

    def test_get_leaves_includes_archived(self, graph):
        root, t1, t2 = _done_graph(graph)
        _archive(graph, t1.id)
        leaves = graph.get_leaves(include_archived=True)
        assert {n.id for n in leaves} == {t1.id, t2.id}

    def test_node_summaries_include_archived(self, graph):
        root, t1, t2 = _done_graph(graph)
        _archive(graph, root.id, t1.id, t2.id)
        summaries = _reads.get_node_summaries(graph._conn, include_archived=True)
        assert {s["id"] for s in summaries} == {root.id, t1.id, t2.id}

    def test_harvest_sees_archived_done_subtree(self, graph):
        # Acceptance: harvest walks include archived DONE goals (wiki/github
        # exporters must not regress when a finished goal is shelved).
        root, t1, t2 = _done_graph(graph)
        graph.mark_running(root.id)
        graph.mark_done(root.id)
        _archive(graph, root.id, t1.id, t2.id)
        summary = build_harvest_summary(graph, graph.get_node(root.id))
        assert summary.status == "done"
        assert summary.tasks_done == 2
        assert summary.tasks_failed == 0


class TestArchivedDisplayMarker:
    """Acceptance #10: include_archived=True renders archived nodes marked."""

    def test_format_node_marks_archived(self, graph):
        root, t1, _ = _done_graph(graph)
        _archive(graph, t1.id)
        node = _reads.get_node(graph._conn, t1.id)
        assert "[archived]" in format_node(node)

    def test_format_node_unmarked_when_active(self, graph):
        root, t1, _ = _done_graph(graph)
        assert "[archived]" not in format_node(graph.get_node(t1.id))

    def test_render_tree_marks_archived_children(self, graph):
        root, t1, t2 = _done_graph(graph)
        _archive(graph, t1.id)
        output = render_tree(graph, include_archived=True)
        assert "Task 1" in output
        assert "[archived]" in output
        # The live sibling renders without the marker.
        lines = {line.strip() for line in output.splitlines()}
        marked = [line for line in lines if "[archived]" in line]
        assert any("Task 1" in line for line in marked)
        assert all("Task 2" not in line for line in marked)

    def test_render_tree_marks_archived_root(self, graph):
        root, t1, t2 = _done_graph(graph)
        _archive(graph, root.id, t1.id, t2.id)
        output = render_tree(graph, include_archived=True)
        assert "Root goal" in output
        assert "[archived]" in output
        assert "2/3" in output

    def test_summarize_counts_archived_when_included(self, graph):
        root, t1, t2 = _done_graph(graph)
        _archive(graph, root.id, t1.id, t2.id)
        s = summarize(graph, include_archived=True)
        assert s.total == 3
        assert s.done == 2


class TestNextRunnableSkipsArchived:
    """todo_next backing read: an archived candidate that would otherwise
    be next (lowest id) must be passed over."""

    def test_next_runnable_skips_lower_id_archived_candidate(self, graph):
        root = graph.add_node("Root")
        shelved = graph.add_node("Shelved first", parent_id=root.id)
        live = graph.add_node("Live second", parent_id=root.id)
        assert shelved.id < live.id  # archived candidate sorts first by id
        _archive(graph, shelved.id)
        nxt = graph.get_next_runnable()
        assert nxt is not None
        assert nxt.id == live.id

    def test_next_runnable_none_when_only_candidate_archived(self, graph):
        root = graph.add_node("Root")
        shelved = graph.add_node("Shelved only", parent_id=root.id)
        _archive(graph, shelved.id)
        assert graph.get_next_runnable() is None


class TestSummariesCombinedFilters:
    """graph_summary backing read: archived filter composes with the
    status/kind/flavor filters instead of replacing them."""

    def test_archived_excluded_under_status_filter(self, graph):
        root, t1, t2 = _done_graph(graph)
        _archive(graph, t1.id)
        done = graph.get_node_summaries(status=NodeStatus.DONE)
        assert [s["id"] for s in done] == [t2.id]

    def test_include_archived_restores_under_status_filter(self, graph):
        root, t1, t2 = _done_graph(graph)
        _archive(graph, t1.id)
        done = _reads.get_node_summaries(
            graph._conn, status=NodeStatus.DONE, include_archived=True
        )
        assert {s["id"] for s in done} == {t1.id, t2.id}

    def test_archived_excluded_under_kind_and_flavor_filters(self, graph):
        root, t1, t2 = _done_graph(graph)
        graph._conn.execute("UPDATE nodes SET flavor = 'spec' WHERE id = ?", (t1.id,))
        graph._conn.execute("UPDATE nodes SET flavor = 'implement' WHERE id = ?", (t2.id,))
        graph._conn.commit()
        _archive(graph, t1.id)
        rows = graph.get_node_summaries(kind=NodeKind.TASK, flavor="spec")
        assert rows == []
        restored = _reads.get_node_summaries(
            graph._conn, kind=NodeKind.TASK, flavor="spec", include_archived=True
        )
        assert [s["id"] for s in restored] == [t1.id]

    def test_status_filter_still_excludes_mismatched_archived(self, graph):
        # include_archived=True widens only the archive axis; a PENDING
        # archived node must not leak into a DONE-filtered summary.
        root = graph.add_node("Root")
        pending = graph.add_node("Pending shelved", parent_id=root.id)
        _archive(graph, pending.id)
        done = _reads.get_node_summaries(
            graph._conn, status=NodeStatus.DONE, include_archived=True
        )
        assert done == []

    def test_summarize_uses_entire_forest(self, graph):
        done = graph.add_node("Done root")
        graph.add_node("Pending root")
        graph.mark_running(done.id)
        graph.mark_done(done.id)

        summary = summarize(graph)
        assert (summary.total, summary.done) == (2, 1)


class TestEmptyAndNoArchiveRegressions:
    """include_archived on an empty DB or a DB with no archived rows must
    behave exactly like the default projection."""

    def test_empty_db_all_projections_empty(self, graph):
        assert _reads.get_all_nodes(graph._conn, include_archived=True) == []
        assert _reads.get_roots(graph._conn, include_archived=True) == []
        assert _reads.get_root(graph._conn, include_archived=True) is None
        assert _reads.get_leaves(graph._conn, include_archived=True) == []
        assert _reads.get_ready_nodes(graph._conn, include_archived=True) == []
        assert _reads.get_node_summaries(graph._conn, include_archived=True) == []
        assert _reads.get_children_map(graph._conn, include_archived=True) == {}
        s = summarize(graph, include_archived=True)
        assert s.total == 0 and s.done == 0

    def test_no_archived_rows_include_archived_matches_default(self, graph):
        _done_graph(graph)
        default = graph.get_all_nodes()
        included = _reads.get_all_nodes(graph._conn, include_archived=True)
        assert [n.id for n in included] == [n.id for n in default]
        default_map = graph.get_children_map()
        included_map = _reads.get_children_map(graph._conn, include_archived=True)
        assert {p: [n.id for n in kids] for p, kids in included_map.items()} == {
            p: [n.id for n in kids] for p, kids in default_map.items()
        }
        assert summarize(graph, include_archived=True).total == summarize(graph).total
        output_default = render_tree(graph)
        output_included = render_tree(graph, include_archived=True)
        assert "[archived]" not in output_default
        assert "[archived]" not in output_included


class TestMarkerExactFormat:
    """Pin the literal marker format (acceptance #10): rich-escaped suffix."""

    def test_format_node_marker_exact_suffix(self, graph):
        root, t1, _ = _done_graph(graph)
        _archive(graph, t1.id)
        node = graph.get_node(t1.id)
        assert format_node(node).endswith(" [dim]\\[archived][/dim]")

    def test_format_node_running_worktree_marker_after_path(self, graph):
        # Marker composes with the worktree suffix instead of clobbering it.
        root, t1, _ = _done_graph(graph)
        graph._conn.execute("UPDATE nodes SET worktree_path = '/tmp/wt' WHERE id = ?", (t1.id,))
        graph._conn.execute("UPDATE nodes SET status = 'running' WHERE id = ?", (t1.id,))
        graph._conn.commit()
        _archive(graph, t1.id)
        node = graph.get_node(t1.id)
        label = format_node(node)
        assert "(/tmp/wt)" in label
        assert label.endswith(" [dim]\\[archived][/dim]")


class TestHarvestFlipsIncludeArchived:
    """The harvest forest walk must pass include_archived=True at the call
    seam (asserted on the seam, not by re-implementing the walk)."""

    def test_harvest_walk_passes_include_archived_true(self, graph, monkeypatch):
        root, t1, t2 = _done_graph(graph)
        calls: list[bool] = []
        real_get_all_nodes = _reads.get_all_nodes

        def spy(conn, *, include_archived=False):
            calls.append(include_archived)
            return real_get_all_nodes(conn, include_archived=include_archived)

        monkeypatch.setattr(_reads, "get_all_nodes", spy)
        build_harvest_summary(graph, graph.get_node(root.id))
        assert calls, "harvest never walked the forest"
        assert all(flag is True for flag in calls)


class TestClaimedAndArchivedMixing:
    """_reads has no claim filter; claims only enrich get_node with
    goal_run_id. An archived claimed goal stays a faithful point lookup."""

    def test_get_node_on_archived_claimed_goal_keeps_claim(self, graph):
        root, t1, t2 = _done_graph(graph)
        graph._conn.execute(
            "UPDATE nodes SET kind = 'goal', flavor = NULL WHERE id = ?", (root.id,)
        )
        graph._conn.execute(
            "INSERT INTO goal_claims (goal_id, run_id, pid, claimed_at) VALUES (?, ?, ?, ?)",
            (root.id, "run-123", 9999, "2026-01-01T00:00:00+00:00"),
        )
        graph._conn.commit()
        _archive(graph, root.id, t1.id, t2.id)
        node = graph.get_node(root.id)
        assert node is not None
        assert node.archived_at is not None
        assert node.goal_run_id == "run-123"
        # ...while the forest projection still hides it.
        assert graph.get_all_nodes() == []
