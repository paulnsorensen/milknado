from __future__ import annotations

from dataclasses import fields, replace
from datetime import UTC, datetime
from typing import cast

import pytest

from milknado.app.graph_view import (
    GraphTreeEntry,
    detail_navigation_text,
    node_inspector_text,
    project_graph,
    validate_local_artifact_path,
)
from milknado.domains.common import (
    MikadoEdge,
    MikadoNode,
    NodeKind,
    NodeStatus,
    SessionContext,
    SessionEvent,
    SessionView,
)
from milknado.domains.graph import (
    ArtifactSnapshot,
    CommandReceipt,
    GraphSnapshot,
    NodeDetailSnapshot,
    NodeSessionSnapshot,
    SnapshotPage,
    SnapshotValue,
)
from milknado.domains.graph._run_persistence import NodeReviewRecord, RunRecord

_CREATED = datetime(2026, 9, 12, tzinfo=UTC)


def test_project_graph_keeps_primary_tree_and_dag_references() -> None:
    root = MikadoNode(1, "root", kind=NodeKind.GOAL, created_at=_CREATED)
    child = MikadoNode(2, "primary child", parent_id=1, created_at=_CREATED)
    other_root = MikadoNode(3, "other root", kind=NodeKind.GOAL, created_at=_CREATED)
    snapshot = GraphSnapshot(
        nodes=(root, child, other_root),
        edges=(MikadoEdge(1, 2), MikadoEdge(3, 2)),
        root_ids=(1, 3),
    )

    projection = project_graph(snapshot)

    assert projection.roots == (GraphTreeEntry(1), GraphTreeEntry(3))
    assert projection.children[GraphTreeEntry(1)] == (GraphTreeEntry(2),)
    assert projection.children[GraphTreeEntry(3)] == (GraphTreeEntry(2, reference_parent_id=3),)
    assert projection.children[GraphTreeEntry(3)][0].is_reference


def test_node_inspector_discloses_every_node_field_and_value() -> None:
    description = "First line of the complete node brief.\nSecond line remains visible."
    node = MikadoNode(
        9,
        description,
        status=NodeStatus.RUNNING,
        kind=NodeKind.TASK,
        flavor="implement",
        artifact_path="docs/node.md",
        created_at=_CREATED,
    )

    rendered = node_inspector_text(node)

    assert "Identity" in rendered
    assert "Execution" in rendered
    assert "Paths" in rendered
    assert "Time" in rendered
    assert (
        "description:\n  First line of the complete node brief.\nSecond line remains visible."
        in rendered
    )
    for node_field in fields(node):
        value = cast(object, getattr(node, node_field.name))
        if node_field.name == "description":
            continue
        if value is None:
            expected = "—"
        elif isinstance(value, (NodeKind, NodeStatus)):
            expected = value.value
        else:
            expected = str(value)
        assert f"{node_field.name}: {expected}" in rendered


def test_validate_local_artifact_path_rejects_escape_and_control_text() -> None:
    assert validate_local_artifact_path("docs/node.md") == "docs/node.md"
    with pytest.raises(ValueError, match="repository-relative"):
        _ = validate_local_artifact_path("../outside.md")
    with pytest.raises(ValueError, match="plain text"):
        _ = validate_local_artifact_path("docs/\x1b[31mnode.md")


def _detail_fixture() -> tuple[MikadoNode, NodeDetailSnapshot]:
    node = MikadoNode(9, "selected node", kind=NodeKind.TASK, status=NodeStatus.RUNNING)
    child = MikadoNode(10, "related child", parent_id=9, created_at=_CREATED)
    session = NodeSessionSnapshot(
        "history-run",
        SessionView(
            context=SessionContext(family="codex", cwd="/repo"),
            events=(SessionEvent(kind="assistant", text="current event"),),
        ),
        "loaded",
        SnapshotPage(
            (SessionEvent(kind="tool", text="older event"),),
            0,
            1,
            2,
            True,
        ),
    )
    run_record: RunRecord = {
        "run_id": "run-42",
        "node_id": 9,
        "status": "failed",
        "pid": None,
        "log_path": "run.log",
        "started_at": "2026-09-12T00:00:00+00:00",
        "ended_at": "2026-09-12T00:01:00+00:00",
        "timed_out": False,
        "exit_code": 1,
        "error": None,
        "timeout_seconds": 30,
        "detail": "failed",
        "rebased": None,
    }
    review_record: NodeReviewRecord = {
        "node_id": 9,
        "round": 1,
        "verdict": "approve",
        "findings": "none",
        "created_at": "2026-09-12T00:02:00+00:00",
    }
    detail = NodeDetailSnapshot(
        node=node,
        description=node.description,
        parent=None,
        children=SnapshotPage((child,), 1, 1, 3, True),
        ancestors=SnapshotPage((), 0, 1, 0, False),
        prerequisite_ids=SnapshotPage((11,), 0, 1, 1, False),
        dependent_ids=SnapshotPage((12,), 0, 1, 1, False),
        reverse_dependents=SnapshotPage((), 0, 1, 0, False),
        owned_files=SnapshotPage(("src/related.py",), 0, 1, 1, False),
        runs=SnapshotPage((run_record,), 0, 1, 1, False),
        reviews=SnapshotPage((review_record,), 0, 1, 1, False),
        sessions=SnapshotPage((session,), 0, 1, 1, False),
        receipts=SnapshotPage(
            (
                CommandReceipt(
                    command_id="cmd-42",
                    status="expired",
                    node_id=9,
                    run_id="run-42",
                    invocation_id="invoke-1",
                    owner_incarnation="owner-1",
                    action="steer",
                    text="safe guidance",
                    permission_id=None,
                    expires_at="2026-09-12T00:00:00+00:00",
                    admitted_at="2026-09-11T23:59:00+00:00",
                    recorded_at="2026-09-12T00:00:01+00:00",
                    detail="expired",
                ),
            ),
            0,
            1,
            1,
            False,
        ),
        goal_claim=SnapshotValue(None, "not_stored"),
        artifacts=SnapshotPage(
            (ArtifactSnapshot("docs/result.md", SnapshotValue("artifact body", "loaded")),),
            0,
            1,
            1,
            False,
        ),
    )
    return node, detail


@pytest.mark.parametrize("unsafe_path", ["/tmp/result.md", "../result.md", "docs/\x00result.md"])
def test_related_artifact_paths_are_validated_before_display(unsafe_path: str) -> None:
    node, detail = _detail_fixture()
    invalid_artifact = ArtifactSnapshot(unsafe_path, SnapshotValue("artifact body", "loaded"))
    detail = replace(detail, artifacts=SnapshotPage((invalid_artifact,), 0, 1, 1, False))

    rendered = node_inspector_text(node, detail)

    assert unsafe_path not in rendered
    assert "<invalid:" in rendered


def test_node_inspector_discloses_related_values_and_history_pages() -> None:
    node, detail = _detail_fixture()

    rendered = node_inspector_text(node, detail)

    assert "children: 2-2/3 loaded (loaded, more available)" in rendered
    assert "related child" in rendered
    assert "run_id: run-42" in rendered
    assert "failed" in rendered
    assert "src/related.py" in rendered
    assert "artifact body" in rendered
    assert "older event" in rendered
    assert "goal_claim: not_stored" in rendered
    assert "receipts: 1-1/1 loaded" in rendered
    assert "command_id: cmd-42" in rendered
    assert "status: expired" in rendered
    assert "text: safe guidance" in rendered
    assert "[ previous" in detail_navigation_text(detail)
    assert "] next" in detail_navigation_text(detail)
    assert ") next" in detail_navigation_text(detail)


def test_detail_navigation_uses_aggregate_page_sets() -> None:
    node, detail = _detail_fixture()
    short_children = SnapshotPage((), 0, 1, 0, False)
    later_receipts = SnapshotPage(detail.receipts.items, 0, 1, 2, True)
    assert detail.sessions.items is not None
    first_session = detail.sessions.items[0]
    later_session = replace(
        first_session,
        run_id="history-run-2",
        event_history=SnapshotPage((), 0, 1, 2, True),
    )
    detail = replace(
        detail,
        children=short_children,
        receipts=later_receipts,
        sessions=SnapshotPage((first_session, later_session), 0, 1, 2, False),
    )

    navigation = detail_navigation_text(detail)

    assert node.id == 9
    assert "] next" in navigation
    assert ") next" in navigation
