from __future__ import annotations

import sqlite3
from dataclasses import fields
from datetime import UTC, datetime
from pathlib import Path

import pytest

from milknado.domains.common import (
    MikadoNode,
    NodeKind,
    NodeSpec,
    NodeStatus,
    SessionContext,
    SessionEvent,
)
from milknado.domains.graph import MikadoGraph, SnapshotPage, read_observer_snapshot
from milknado.domains.graph.snapshot import connect_readonly
from milknado.domains.graph.snapshot_history import ancestors as ancestor_chain
from milknado.domains.graph.snapshot_history import values_page


def _complete_node(graph: MikadoGraph, db_path: Path) -> int:
    node = graph.add_node(
        "complete description",
        spec=NodeSpec(
            kind=NodeKind.GOAL,
            wiki_ref="wiki-root",
            github_ref="PVTI-root",
            artifact_path="docs/root.md",
            oversized=True,
            batch_index=4,
        ),
    )
    _ = graph.claim_or_reclaim_goal(node.id, "goal-run", pid=123, now="2026-09-11T00:00:00+00:00")
    _ = graph.files.claim(node.id, ["src/root.py", "tests/root.py"])
    _ = graph.runs.start(
        "node-run",
        node.id,
        str(db_path.parent / "run.log"),
        "2026-09-11T00:00:01+00:00",
        60,
        pid=456,
    )
    _ = graph.sessions.start("node-run", SessionContext(family="codex", cwd=str(db_path.parent)))
    _ = graph.runs.insert_review(node.id, "approve", "clean", "2026-09-11T00:00:02+00:00")
    _ = graph.sessions.append(
        "node-run",
        SessionEvent(kind="status", text="running"),
    )
    _ = graph._conn.execute(  # pyright: ignore[reportPrivateUsage]
        "UPDATE nodes SET status = ?, worktree_path = ?, branch_name = ?, run_id = ?, pid = ?, "
        + "created_at = ?, completed_at = ?, dispatched_at = ?, "
        + "completion_duration_seconds = ? WHERE id = ?",
        (
            NodeStatus.RUNNING.value,
            "/tmp/worktree",
            "feature/root",
            "node-run",
            456,
            "2026-09-11T00:00:00+00:00",
            "2026-09-11T00:00:03+00:00",
            "2026-09-11T00:00:01+00:00",
            2.5,
            node.id,
        ),
    )
    _ = graph._conn.commit()  # pyright: ignore[reportPrivateUsage]
    return node.id


def test_node_detail_covers_every_mikado_node_field_and_related_history(tmp_path: Path) -> None:
    db_path = tmp_path / "graph.db"
    graph = MikadoGraph(db_path)
    root_id = _complete_node(graph, db_path)
    child = graph.add_node("child", parent_id=root_id)
    sibling = graph.add_node("sibling", parent_id=root_id)
    response = graph.get_node_detail_snapshot(root_id, request_generation=8, limit=1)
    assert response.matches(root_id, 8)
    assert response.detail is not None
    detail = response.detail
    assert {field.name for field in fields(detail.node)} == {
        field.name for field in fields(MikadoNode)
    }
    assert detail.node == MikadoNode(
        id=root_id,
        description="complete description",
        status=NodeStatus.RUNNING,
        worktree_path="/tmp/worktree",
        branch_name="feature/root",
        run_id="node-run",
        pid=456,
        created_at=datetime(2026, 9, 11, tzinfo=UTC),
        completed_at=datetime(2026, 9, 11, 0, 0, 3, tzinfo=UTC),
        dispatched_at=datetime(2026, 9, 11, 0, 0, 1, tzinfo=UTC),
        oversized=True,
        batch_index=4,
        completion_duration_seconds=2.5,
        kind=NodeKind.GOAL,
        flavor=None,
        goal_run_id="goal-run",
        wiki_ref="wiki-root",
        github_ref="PVTI-root",
        artifact_path="docs/root.md",
    )
    assert detail.description == detail.node.description
    assert {node.id for node in detail.children.items or ()} == {child.id}
    assert detail.children.has_more is True
    assert detail.prerequisite_ids.items == (child.id,)
    assert detail.dependent_ids.items == ()
    assert detail.owned_files.items == ("src/root.py",)
    assert detail.runs.items is not None and detail.runs.items[0]["run_id"] == "node-run"
    assert detail.reviews.items is not None and detail.reviews.items[0]["verdict"] == "approve"
    assert detail.sessions.items is not None and detail.sessions.items[0].state == "loaded"
    assert detail.goal_claim.value is not None
    assert detail.goal_claim.value["run_id"] == "goal-run"
    assert detail.artifacts.items is not None
    assert detail.artifacts.items[0].content.state == "not_loaded"
    child_detail = graph.get_node_detail_snapshot(child.id, limit=10).detail
    assert child_detail is not None
    assert child_detail.dependent_ids.items == (root_id,)
    assert child_detail.reverse_dependents.items is not None
    assert child_detail.reverse_dependents.items[0].id == root_id
    assert sibling.id in {node.id for node in graph.get_graph_snapshot().nodes}
    graph.close()


def test_observer_snapshot_uses_readonly_connection_and_response_fences(tmp_path: Path) -> None:
    db_path = tmp_path / "graph.db"
    graph = MikadoGraph(db_path)
    node_id = graph.add_node("root").id
    graph.close()

    connection = connect_readonly(db_path)
    assert connection.execute("PRAGMA query_only").fetchone()[0] == 1
    with pytest.raises(sqlite3.OperationalError):
        _ = connection.execute("CREATE TABLE forbidden_write (id INTEGER)")
    connection.close()
    snapshot = read_observer_snapshot(db_path, node_id=node_id, request_generation=11)
    assert snapshot.graph is not None
    assert snapshot.node is not None
    assert snapshot.node.matches(node_id, 11)
    assert not snapshot.node.matches(node_id, 12)
    assert snapshot.node.detail is not None
    missing = read_observer_snapshot(db_path, node_id=999, request_generation=4).node
    assert missing is not None
    assert missing.matches(999, 4)
    assert missing.detail is None


def test_detail_history_pages_are_bounded_and_keep_retained_state(tmp_path: Path) -> None:
    db_path = tmp_path / "graph.db"
    graph = MikadoGraph(db_path)
    root = graph.add_node("root", spec=NodeSpec(artifact_path="docs/root.md"))
    middle = graph.add_node("middle", parent_id=root.id)
    leaf = graph.add_node("leaf", parent_id=middle.id)

    assert tuple(item.id for item in ancestor_chain(graph._conn, leaf)) == (middle.id, root.id)  # pyright: ignore[reportPrivateUsage]
    for page_number, expected_ids in ((0, (middle.id,)), (1, (root.id,)), (2, ())):
        detail = graph.get_node_detail_snapshot(leaf.id, page=page_number, limit=1).detail
        assert detail is not None
        ancestors = detail.ancestors
        assert tuple(node.id for node in ancestors.items or ()) == expected_ids
        assert ancestors.total == 2
        assert ancestors.has_more == (page_number == 0)
        assert ancestors.state == "loaded"

    root_detail = graph.get_node_detail_snapshot(root.id, page=1, limit=1).detail
    assert root_detail is not None
    assert root_detail.children.items == ()
    assert root_detail.children.total == 1
    assert root_detail.children.state == "loaded"
    artifacts = graph.get_node_detail_snapshot(root.id, limit=1).detail
    assert artifacts is not None
    assert artifacts.artifacts.items is not None
    assert artifacts.artifacts.items[0].content.state == "not_loaded"
    overflow = graph.get_node_detail_snapshot(root.id, page=5, limit=1).detail
    assert overflow is not None
    assert overflow.artifacts.items == ()
    assert overflow.artifacts.total == 1
    assert overflow.artifacts.state == "loaded"

    not_retained: SnapshotPage[object] = values_page(
        [], page=5, limit=1, total=None, state="not_retained"
    )
    assert not_retained.items is None
    assert not_retained.state == "not_retained"
    graph.close()


def test_detail_dag_references_hide_archived_nodes_consistently(tmp_path: Path) -> None:
    db_path = tmp_path / "graph.db"
    graph = MikadoGraph(db_path)
    root = graph.add_node("root")
    active = graph.add_node("active", parent_id=root.id)
    archived_child = graph.add_node("archived child", parent_id=root.id)
    archived_parent = graph.add_node("archived parent")
    _ = graph._conn.execute(  # pyright: ignore[reportPrivateUsage]
        "INSERT INTO edges(parent_id, child_id) VALUES (?, ?)", (archived_parent.id, active.id)
    )
    _ = graph._conn.execute(  # pyright: ignore[reportPrivateUsage]
        "UPDATE nodes SET archived_at = ? WHERE id IN (?, ?)",
        ("2026-09-12T00:00:00+00:00", archived_child.id, archived_parent.id),
    )
    _ = graph._conn.commit()  # pyright: ignore[reportPrivateUsage]

    root_detail = graph.get_node_detail_snapshot(root.id, limit=10).detail
    assert root_detail is not None
    assert [node.id for node in root_detail.children.items or ()] == [active.id]
    assert root_detail.prerequisite_ids.items == (active.id,)
    active_detail = graph.get_node_detail_snapshot(active.id, limit=10).detail
    assert active_detail is not None
    assert active_detail.dependent_ids.items == (root.id,)
    assert active_detail.reverse_dependents.items == (root,)
    graph.close()


def test_graph_snapshot_cache_refreshes_after_graph_mutation(tmp_path: Path) -> None:
    db_path = tmp_path / "graph.db"
    graph = MikadoGraph(db_path)
    first = graph.get_graph_snapshot()

    assert graph.get_graph_snapshot() is first
    added = graph.add_node("new node")
    refreshed = graph.get_graph_snapshot()

    assert refreshed is not first
    assert tuple(node.id for node in refreshed.nodes) == (added.id,)
    graph.close()
