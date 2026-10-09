import os
from pathlib import Path
from typing import cast

from milknado.domains.common import NodeKind, NodeSpec
from milknado.domains.coordinator import ControlEvent, CoordinatorControl
from milknado.domains.coordinator.control_models import StartGoal
from milknado.domains.coordinator.journal import append_control_event
from milknado.domains.graph import MikadoGraph


def _started(tmp_path: Path) -> tuple[MikadoGraph, CoordinatorControl, str, int]:
    graph = MikadoGraph(tmp_path / "graph.db")
    control = CoordinatorControl(graph, tmp_path)
    receipt = control.send_coordinator_command("", StartGoal("start", "Deliver", "codex"))
    result = cast(dict[str, object], receipt.result)
    return graph, control, cast(str, result["id"]), cast(int, result["goal_id"])


def test_diamond_projection_contains_each_node_once(tmp_path: Path) -> None:
    graph, control, session_id, goal_id = _started(tmp_path)
    left = graph.add_node("left", parent_id=goal_id)
    right = graph.add_node("right", parent_id=goal_id)
    shared = graph.add_node("shared", parent_id=left.id)
    _ = graph.add_edge(right.id, shared.id)

    nodes = control.read_coordinator_snapshot(session_id, 0).nodes
    assert nodes[0].id == goal_id
    assert {node.id for node in nodes} == {goal_id, left.id, right.id, shared.id}
    assert len(nodes) == 4
    graph.close()


def test_claimed_descendant_goal_keeps_run_id(tmp_path: Path) -> None:
    graph, control, session_id, goal_id = _started(tmp_path)
    child = graph.add_node("child goal", parent_id=goal_id, spec=NodeSpec(kind=NodeKind.GOAL))
    assert graph.claim_or_reclaim_goal(
        child.id, "child-run", os.getpid(), now="2026-01-01T00:00:00+00:00"
    )

    nodes = control.read_coordinator_snapshot(session_id, 0).nodes
    assert [node.id for node in nodes] == [goal_id, child.id]
    assert nodes[1].goal_run_id == "child-run"
    graph.close()


def test_late_cursor_queries_only_new_events_and_complete_recovery(tmp_path: Path) -> None:
    graph, control, session_id, _ = _started(tmp_path)
    conn = graph.group_connection
    first_recovery = append_control_event(
        conn, session_id, ControlEvent(kind="recovery", text="first")
    )
    for index in range(20):
        _ = append_control_event(conn, session_id, ControlEvent(kind="command", text=str(index)))
    cursor = control.read_coordinator_snapshot(session_id, 0).cursor
    second_recovery = append_control_event(
        conn, session_id, ControlEvent(kind="recovery", text="second")
    )
    latest = append_control_event(conn, session_id, ControlEvent(kind="command", text="latest"))
    queries: list[str] = []
    conn.set_trace_callback(queries.append)
    try:
        snapshot = control.read_coordinator_snapshot(session_id, cursor)
    finally:
        conn.set_trace_callback(None)
    assert [event.seq for event in snapshot.events] == [second_recovery, latest]
    assert [event.seq for event in snapshot.recovery] == [first_recovery, second_recovery]
    assert snapshot.cursor == latest
    event_queries = [query for query in queries if "FROM coordinator_events" in query]
    assert event_queries
    assert all(
        "seq >" in query or "kind = 'recovery'" in query or "MAX(seq)" in query
        for query in event_queries
    )
    assert control.read_coordinator_snapshot(session_id, latest + 10).cursor == latest
    graph.close()


def test_expired_event_is_excluded_from_events_and_cursor(tmp_path: Path) -> None:
    graph, control, session_id, _ = _started(tmp_path)
    cursor = control.read_coordinator_snapshot(session_id, 0).cursor
    expired = append_control_event(
        graph.group_connection, session_id, ControlEvent(kind="diagnostic", text="expired")
    )
    with graph.group_connection:
        _ = graph.group_connection.execute(
            "UPDATE coordinator_events SET expires_at = ? WHERE seq = ?",
            ("2000-01-01T00:00:00+00:00", expired),
        )
    snapshot = control.read_coordinator_snapshot(session_id, cursor)
    assert snapshot.events == ()
    assert snapshot.cursor == cursor
    graph.close()


def test_archived_descendants_are_not_projected(tmp_path: Path) -> None:
    graph, control, session_id, goal_id = _started(tmp_path)
    child = graph.add_node("child", parent_id=goal_id)
    assert [node.id for node in control.read_coordinator_snapshot(session_id, 0).nodes] == [
        goal_id,
        child.id,
    ]
    graph.mark_running(child.id)
    graph.mark_done(child.id)
    _ = graph.archive_subtree(child.id)
    assert [node.id for node in control.read_coordinator_snapshot(session_id, 0).nodes] == [
        goal_id
    ]
    graph.close()
