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
from milknado.domains.graph import (
    MikadoGraph,
    read_node_detail_snapshot,
    read_observer_snapshot,
)


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
        "id",
        "description",
        "status",
        "parent_id",
        "worktree_path",
        "branch_name",
        "run_id",
        "pid",
        "created_at",
        "completed_at",
        "dispatched_at",
        "oversized",
        "batch_index",
        "completion_duration_seconds",
        "kind",
        "flavor",
        "goal_run_id",
        "wiki_ref",
        "github_ref",
        "artifact_path",
        "archived_at",
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
    assert detail.artifacts.items[0].content.state == "not_stored"
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

    connection = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)
    with pytest.raises(sqlite3.OperationalError):
        _ = connection.execute("CREATE TABLE forbidden_write (id INTEGER)")
    connection.close()

    snapshot = read_observer_snapshot(db_path, node_id=node_id, request_generation=11)
    assert snapshot.graph is not None
    assert snapshot.node is not None
    assert snapshot.node.matches(node_id, 11)
    assert not snapshot.node.matches(node_id, 12)
    assert snapshot.node.detail is not None
    missing = read_node_detail_snapshot(db_path, 999, request_generation=4)
    assert missing.matches(999, 4)
    assert missing.detail is None
