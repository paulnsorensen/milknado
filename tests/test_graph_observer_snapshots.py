from __future__ import annotations

import os
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from milknado.app.watch import WatchSnapshotSource
from milknado.domains.common import (
    MikadoNode,
    NodeKind,
    NodeSpec,
    NodeStatus,
    SessionContext,
    SessionEvent,
)
from milknado.domains.graph import MikadoGraph, read_observer_snapshot
from milknado.domains.graph import snapshot_history as history
from milknado.domains.graph.snapshot import connect_readonly


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
    root = graph.add_node("root")
    children = tuple(graph.add_node(f"child-{index}", parent_id=root.id) for index in range(2))
    node_id = root.id
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
    assert tuple(node.id for node in snapshot.node.detail.children.items or ()) == tuple(
        child.id for child in children
    )
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
    _ = graph.files.claim(root.id, ["a.py", "b.py", "c.py"])
    for index in range(3):
        _ = graph.runs.start(
            f"run-{index}",
            root.id,
            str(tmp_path / f"run-{index}.log"),
            f"2026-09-12T00:00:0{index}+00:00",
            60,
        )

    for page_number, expected_ids in ((0, (middle.id,)), (1, (root.id,)), (2, ())):
        detail = graph.get_node_detail_snapshot(leaf.id, page=page_number, limit=1).detail
        assert detail is not None
        ancestors = detail.ancestors
        assert tuple(node.id for node in ancestors.items or ()) == expected_ids
        assert ancestors.total == 2
        assert ancestors.has_more == (page_number == 0)
        assert ancestors.state == "loaded"

    owned_files: list[str] = []
    run_ids: list[str] = []
    for page_number in range(3):
        detail = graph.get_node_detail_snapshot(root.id, page=page_number, limit=1).detail
        assert detail is not None
        owned_files.extend(detail.owned_files.items or ())
        run_ids.extend(run["run_id"] for run in detail.runs.items or ())
        if not detail.owned_files.has_more and not detail.runs.has_more:
            break

    assert tuple(owned_files) == ("a.py", "b.py", "c.py")
    assert tuple(run_ids) == ("run-2", "run-1", "run-0")

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
    graph.close()


def test_snapshot_states_distinguish_missing_session_rows_from_missing_storage(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "graph.db"
    graph = MikadoGraph(db_path)
    node = graph.add_node("node", spec=NodeSpec(artifact_path="docs/node.md"))
    _ = graph.runs.start(
        "run-without-session",
        node.id,
        str(tmp_path / "run.log"),
        "2026-09-12T00:00:00+00:00",
        60,
    )

    detail = graph.get_node_detail_snapshot(node.id, limit=1).detail
    assert detail is not None
    assert detail.sessions.items is not None
    assert detail.sessions.items[0].state == "missing"
    assert detail.sessions.items[0].session is None
    assert detail.sessions.items[0].event_history.state == "missing"
    assert detail.goal_claim.state == "loaded"
    assert detail.goal_claim.value is None
    assert detail.artifacts.items is not None
    assert detail.artifacts.items[0].content.state == "not_loaded"

    _ = graph._conn.execute("DROP TABLE run_messages")  # pyright: ignore[reportPrivateUsage]
    _ = graph._conn.execute("DROP TABLE run_sessions")  # pyright: ignore[reportPrivateUsage]
    _ = graph._conn.commit()  # pyright: ignore[reportPrivateUsage]
    not_stored = graph.get_node_detail_snapshot(node.id, limit=1).detail
    assert not_stored is not None
    assert not_stored.sessions.state == "not_stored"
    graph.close()


def test_detail_dag_references_hide_archived_nodes_consistently(tmp_path: Path) -> None:
    db_path = tmp_path / "graph.db"
    graph = MikadoGraph(db_path)
    root = graph.add_node("root")
    active = graph.add_node("active", parent_id=root.id)
    archived_child = graph.add_node("archived child", parent_id=root.id)
    archived_parent = graph.add_node("archived parent")
    archived_id = archived_child.id
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
    assert graph.get_node_detail_snapshot(archived_id, limit=10).detail is None
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


def test_graph_snapshot_roots_follow_parent_identity_not_dag_edges(tmp_path: Path) -> None:
    db_path = tmp_path / "graph.db"
    graph = MikadoGraph(db_path)
    root = graph.add_node("root")
    child = graph.add_node("child", parent_id=root.id)
    detached = graph.add_node("detached")
    _ = graph._conn.execute(  # pyright: ignore[reportPrivateUsage]
        "INSERT INTO edges(parent_id, child_id) VALUES (?, ?)", (detached.id, child.id)
    )
    _ = graph._conn.commit()  # pyright: ignore[reportPrivateUsage]

    snapshot = graph.get_graph_snapshot()

    assert snapshot.root_ids == (root.id, detached.id)
    assert {(edge.parent_id, edge.child_id) for edge in snapshot.edges} == {
        (root.id, child.id),
        (detached.id, child.id),
    }
    detail = graph.get_node_detail_snapshot(root.id, limit=10).detail
    assert detail is not None
    assert tuple(node.id for node in detail.children.items or ()) == (child.id,)
    assert detail.prerequisite_ids.items == (child.id,)
    graph.close()


def test_snapshot_page_rejects_zero_limit(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    node = graph.add_node("node")

    with pytest.raises(ValueError, match="limit must be between 1 and 100"):
        _ = graph.get_node_detail_snapshot(node.id, limit=0)
    graph.close()


def test_graph_snapshot_reads_one_cross_connection_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "graph.db"
    reader = MikadoGraph(db_path)
    writer = MikadoGraph(db_path)
    root = reader.add_node("root")
    child = reader.add_node("child", parent_id=root.id)
    detached = reader.add_node("detached")
    original_hydrate = history.hydrate
    inserted = False

    def insert_edge_after_nodes(conn: sqlite3.Connection, rows: list[sqlite3.Row]):
        nonlocal inserted
        if not inserted:
            inserted = True
            _ = writer._conn.execute(  # pyright: ignore[reportPrivateUsage]
                "INSERT INTO edges(parent_id, child_id) VALUES (?, ?)", (detached.id, child.id)
            )
            _ = writer._conn.commit()  # pyright: ignore[reportPrivateUsage]
        return original_hydrate(conn, rows)

    monkeypatch.setattr(history, "hydrate", insert_edge_after_nodes)
    try:
        snapshot = reader.get_graph_snapshot()
        assert (detached.id, child.id) not in {
            (edge.parent_id, edge.child_id) for edge in snapshot.edges
        }
    finally:
        reader.close()
        writer.close()


def test_session_snapshot_pages_normalized_event_history(tmp_path: Path) -> None:
    db_path = tmp_path / "graph.db"
    graph = MikadoGraph(db_path)
    node = graph.add_node("node")
    _ = graph.runs.start(
        "distinct", node.id, str(tmp_path / "distinct.log"), "2026-09-12T00:00:00+00:00", 60
    )
    _ = graph.sessions.start("distinct", SessionContext(family="codex", cwd=str(tmp_path)))
    for index in range(501):
        _ = graph.sessions.append("distinct", SessionEvent(kind="status", text=str(index)))

    values: list[str] = []
    event_history = None
    for page in range(6):
        detail = graph.get_node_detail_snapshot(node.id, limit=100, session_event_page=page).detail
        assert detail is not None and detail.sessions.items is not None
        event_history = detail.sessions.items[0].event_history
        values.extend(event.text for event in event_history.items or ())
        if not event_history.has_more:
            break

    assert len(values) == 501
    assert len(set(values)) == 501
    assert event_history is not None
    assert event_history.total == 501
    assert event_history.has_more is False

    _ = graph.runs.start(
        "revisions", node.id, str(tmp_path / "revisions.log"), "2026-09-12T00:01:00+00:00", 60
    )
    _ = graph.sessions.start("revisions", SessionContext(family="codex", cwd=str(tmp_path)))
    for index in range(501):
        _ = graph.sessions.append(
            "revisions",
            SessionEvent(kind="status", text=str(index), event_id="same"),
        )
    detail = graph.get_node_detail_snapshot(node.id, limit=100).detail
    assert detail is not None and detail.sessions.items is not None
    assert detail.sessions.items[0].event_history.total == 1
    assert detail.sessions.items[0].event_history.has_more is False
    graph.close()


def test_node_detail_snapshot_reads_one_cross_connection_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "graph.db"
    reader = MikadoGraph(db_path)
    writer = MikadoGraph(db_path)
    root = reader.add_node("root")
    detached = reader.add_node("detached")
    original_ancestor_page = history.ancestor_page
    inserted = False

    def insert_edge_after_ancestors(
        conn: sqlite3.Connection, node: MikadoNode, page: int, limit: int
    ):
        nonlocal inserted
        if not inserted:
            inserted = True
            _ = writer._conn.execute(  # pyright: ignore[reportPrivateUsage]
                "INSERT INTO edges(parent_id, child_id) VALUES (?, ?)", (root.id, detached.id)
            )
            _ = writer._conn.commit()  # pyright: ignore[reportPrivateUsage]
        return original_ancestor_page(conn, node, page, limit)

    monkeypatch.setattr(history, "ancestor_page", insert_edge_after_ancestors)
    try:
        detail = reader.get_node_detail_snapshot(root.id, limit=10).detail
        assert detail is not None
        assert detail.prerequisite_ids.items == ()
    finally:
        reader.close()
        writer.close()


def test_watch_graph_cache_ignores_run_writes_and_refreshes_claims(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "graph.db"
    writer = MikadoGraph(db_path)
    goal = writer.add_node("goal", spec=NodeSpec(kind=NodeKind.GOAL))
    for index in range(100):
        _ = writer.add_node(f"child-{index}", parent_id=goal.id)
    traces: list[str] = []

    def connect_with_trace(path: Path) -> sqlite3.Connection:
        connection = connect_readonly(path)
        connection.set_trace_callback(traces.append)
        return connection

    with patch("milknado.app.watch.connect_readonly", connect_with_trace):
        source = WatchSnapshotSource(tmp_path, db_path)
        first = source.snapshot()
        assert first.graph is not None
        traces.clear()
        for _ in range(3):
            _ = source.snapshot()
        writer.runs.start(
            "unrelated-run",
            goal.id,
            str(tmp_path / "run.log"),
            "2026-09-12T00:00:00+00:00",
            60,
        )
        unchanged = source.snapshot()
        hydration_reads = [
            sql
            for sql in traces
            if "SELECT * FROM nodes WHERE archived_at IS NULL" in sql
            or "SELECT parent_id, child_id FROM edges" in sql
        ]
        assert unchanged.graph is first.graph
        assert hydration_reads == []

        _ = writer.claim_or_reclaim_goal(
            goal.id, "goal-run", os.getpid(), now="2026-09-12T00:00:01+00:00"
        )
        refreshed = source.snapshot()
        assert refreshed.graph is not unchanged.graph
        assert refreshed.graph is not None
        assert refreshed.graph.nodes[0].goal_run_id == "goal-run"
        assert any("SELECT * FROM nodes WHERE archived_at IS NULL" in sql for sql in traces)
        source.close()
    writer.close()
