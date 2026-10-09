import sqlite3
from contextlib import closing
from pathlib import Path
from typing import cast

import pytest

from milknado.domains.common import SessionEvent, SessionInput
from milknado.domains.coordinator import CoordinatorControl
from milknado.domains.coordinator.control_models import (
    AttemptCommand,
    CoordinatorCommandReceipt,
    CreateGroup,
    DispatchTask,
    FinishTask,
    Recover,
    RuntimeAction,
    StartGoal,
)
from milknado.domains.coordinator.turn_context import TurnContext
from milknado.domains.coordinator.turns import record_turn_event
from milknado.domains.coordinator.workflow import CoordinatorWorkflow
from milknado.domains.graph import MikadoGraph


def _result(receipt: CoordinatorCommandReceipt) -> dict[str, object]:
    return cast(dict[str, object], receipt.result)


def test_commands_are_durable_and_reject_changed_payloads(tmp_path: Path) -> None:
    path = tmp_path / "graph.db"
    graph = MikadoGraph(path)
    control = CoordinatorControl(graph, tmp_path)
    start = StartGoal("start-1", "Deliver", "codex")
    first = control.send_coordinator_command("", start)
    assert first.status == "accepted"
    assert control.send_coordinator_command("", start) == first
    session_id = cast(str, _result(first)["id"])
    with pytest.raises(ValueError, match="reused"):
        _ = control.send_coordinator_command("", StartGoal("start-1", "Changed", "codex"))
    graph.close()

    reopened = MikadoGraph(path)
    control = CoordinatorControl(reopened, tmp_path)
    assert control.send_coordinator_command("", start) == first
    assert [(item.id, item.description) for item in control.list_coordinator_sessions()] == [
        (session_id, "Deliver")
    ]
    unavailable = control.send_coordinator_command(session_id, Recover("recover-1"))
    assert unavailable.status == "unavailable"
    assert control.send_coordinator_command(session_id, Recover("recover-1")) == unavailable
    with pytest.raises(KeyError):
        _ = control.read_coordinator_snapshot("foreign", 0)
    reopened.close()


def test_snapshot_orders_events_and_links_group_run(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    control = CoordinatorControl(graph, tmp_path)
    start = control.send_coordinator_command("", StartGoal("start-1", "Deliver", "codex"))
    session_id = cast(str, _result(start)["id"])
    task = graph.add_node("Implement", cast(int, _result(start)["goal_id"]), files=("src/a.py",))
    group = control.send_coordinator_command(
        session_id,
        CreateGroup(
            "group-1", "main", (task.id,), str(tmp_path / "group"), "branch", "provider-1"
        ),
    )
    assert group.status == "accepted"
    group_id = cast(str, _result(group)["id"])
    run = control.send_coordinator_command(
        session_id, DispatchTask("dispatch-1", group_id, task.id, "run-1")
    )
    assert run.status == "accepted"
    attempt = cast(dict[str, object], _result(run)["attempt"])
    attempt_id = cast(str, attempt["attempt_id"])
    launched = control.send_coordinator_command(
        session_id, AttemptCommand("launch-1", group_id, task.id, "run-1", attempt_id)
    )
    assert launched.status == "accepted"
    graph.runs.record_verification(attempt_id, True, "2026-01-01T00:00:05+00:00")
    snapshot = control.read_coordinator_snapshot(session_id, 0)
    assert [event.seq for event in snapshot.events] == sorted(
        event.seq for event in snapshot.events
    )
    assert {link.kind for link in snapshot.links} >= {"execution_group", "run", "provider_session"}
    assert snapshot.groups[0].id == group_id
    assert snapshot.runs[0]["run_id"] == attempt_id
    assert snapshot.runs[0]["verification_status"] == "accepted"
    assert snapshot.runs[0]["verified_at"] == "2026-01-01T00:00:05+00:00"
    assert snapshot.cursor == snapshot.events[-1].seq
    assert control.read_coordinator_snapshot(session_id, snapshot.cursor).events == ()
    graph.close()


def test_snapshot_includes_non_parent_goal_dependency(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    control = CoordinatorControl(graph, tmp_path)
    receipt = control.send_coordinator_command("", StartGoal("start", "Deliver", "codex"))
    session_id = cast(str, _result(receipt)["id"])
    goal_id = cast(int, _result(receipt)["goal_id"])
    first = graph.add_node("First", goal_id)
    second = graph.add_node("Second", goal_id)
    _ = graph.add_edge(first.id, second.id)

    snapshot = control.read_coordinator_snapshot(session_id, 0)
    assert {(edge.parent_id, edge.child_id) for edge in snapshot.edges} == {
        (goal_id, first.id),
        (goal_id, second.id),
        (first.id, second.id),
    }
    graph.close()


def test_attempt_lifecycle_through_commands(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    control = CoordinatorControl(graph, tmp_path)
    start = control.send_coordinator_command("", StartGoal("start", "Deliver", "codex"))
    session_id = cast(str, _result(start)["id"])
    task = graph.add_node("Implement", cast(int, _result(start)["goal_id"]))
    group = control.send_coordinator_command(
        session_id,
        CreateGroup("group", "main", (task.id,), str(tmp_path / "group"), "branch", "provider"),
    )
    group_id = cast(str, _result(group)["id"])
    dispatch = control.send_coordinator_command(
        session_id, DispatchTask("dispatch", group_id, task.id, "run")
    )
    attempt = cast(dict[str, object], _result(dispatch)["attempt"])
    attempt_id = cast(str, attempt["attempt_id"])
    launch = control.send_coordinator_command(
        session_id, AttemptCommand("launch", group_id, task.id, "run", attempt_id)
    )
    assert launch.status == "accepted"
    assert _result(launch)["state"] == "launched"
    finish = control.send_coordinator_command(
        session_id, FinishTask("finish", group_id, task.id, "run", attempt_id, True, "verified")
    )
    assert finish.status == "accepted"
    assert graph.groups.task_result(task.id) == ("done", "verified")
    transitions = [
        event.status
        for event in control.read_coordinator_snapshot(session_id, 0).events
        if event.kind == "run_transition"
    ]
    assert transitions == ["claimed", "running", "done"]
    snapshot = control.read_coordinator_snapshot(session_id, 0)
    assert {run["run_id"] for run in snapshot.runs} == {attempt_id}
    assert ("run", attempt_id) in {(link.kind, link.entity_id) for link in snapshot.links}
    assert [event.entity_id for event in snapshot.events if event.kind == "run_transition"] == [
        attempt_id
    ] * 3
    graph.close()


def test_permission_events_keep_request_and_provider_turn_identity(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    control = CoordinatorControl(graph, tmp_path)
    workflow = CoordinatorWorkflow(graph, graph.group_connection)
    session = workflow.start_goal("Deliver", "codex")
    request = SessionEvent(
        kind="permission", text="Approve", event_id="1/shared", state="requested"
    )
    record_turn_event(
        TurnContext(graph.group_connection, session.id, "turn-a"), request, "provider-a"
    )
    first = control.read_coordinator_snapshot(session.id, 0)
    record_turn_event(
        TurnContext(graph.group_connection, session.id, "turn-b"), request, "provider-b"
    )
    delta = control.read_coordinator_snapshot(session.id, first.cursor)
    permissions = [
        event
        for event in control.read_coordinator_snapshot(session.id, 0).events
        if event.kind == "permission"
    ]
    assert [
        (event.entity_id, event.turn_id, event.provider_session_id) for event in permissions
    ] == [
        ("1/shared", "turn-a", "provider-a"),
        ("1/shared", "turn-b", "provider-b"),
    ]
    assert [
        (event.turn_id, event.provider_session_id)
        for event in delta.events
        if event.kind == "permission"
    ] == [("turn-b", "provider-b")]
    graph.close()


def test_missing_runtime_returns_receipt_without_claiming_action(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    control = CoordinatorControl(graph, tmp_path)
    start = control.send_coordinator_command("", StartGoal("start-1", "Deliver", "codex"))
    session_id = cast(str, _result(start)["id"])
    result = control.send_coordinator_command(
        session_id,
        RuntimeAction("action-1", "provider-1", SessionInput(action="interrupt")),
    )
    assert result.status == "unavailable"
    assert (
        control.send_coordinator_command(
            session_id, RuntimeAction("action-1", "provider-1", SessionInput(action="interrupt"))
        )
        == result
    )
    graph.close()


def test_rejected_start_goal_receipt_replays_without_session(tmp_path: Path) -> None:
    path = tmp_path / "graph.db"
    graph = MikadoGraph(path)
    control = CoordinatorControl(graph, tmp_path)
    command = StartGoal("empty-goal", " ", "codex")
    first = control.send_coordinator_command("", command)
    assert first.status == "rejected"
    assert first.session_id == ""
    assert isinstance(first.result, str) and first.result
    assert control.send_coordinator_command("", command) == first
    count = cast(
        tuple[int] | None,
        graph.group_connection.execute("SELECT COUNT(*) FROM coordinator_sessions").fetchone(),
    )
    assert count is not None and count[0] == 0
    graph.close()

    reopened = MikadoGraph(path)
    assert CoordinatorControl(reopened, tmp_path).send_coordinator_command("", command) == first
    reopened.close()


def test_redacted_command_ids_keep_distinct_journal_events(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    control = CoordinatorControl(graph, tmp_path)
    started = control.send_coordinator_command("", StartGoal("start", "Deliver", "codex"))
    session_id = cast(str, _result(started)["id"])
    first = Recover("token=first")
    second = Recover("token=second")
    assert control.send_coordinator_command(session_id, first).status == "unavailable"
    assert control.send_coordinator_command(session_id, second).status == "unavailable"
    assert control.send_coordinator_command(session_id, first).status == "unavailable"
    events = [
        event
        for event in control.read_coordinator_snapshot(session_id, 0).events
        if event.kind == "command" and event.text == "Recover"
    ]
    assert len(events) == 2
    assert len({event.entity_id for event in events}) == 2
    assert all(
        "first" not in event.entity_id and "second" not in event.entity_id for event in events
    )
    graph.close()


def test_coordinator_operation_tables_exist_before_use_and_after_reopen(tmp_path: Path) -> None:

    path = tmp_path / "graph.db"
    for _ in range(2):
        graph = MikadoGraph(path)
        with closing(sqlite3.connect(path)) as conn:
            tables = {
                row[0]
                for row in cast(
                    "list[tuple[str, ...]]",
                    conn.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'table' "
                        + "AND name LIKE 'coordinator_%'"
                    ).fetchall(),
                )
            }
            expected = {
                "coordinator_dispatches",
                "coordinator_action_receipts",
                "coordinator_plans",
            }
            assert expected <= tables
            for table in expected:
                keys = cast(
                    "list[tuple[object, object, object, object, object, int]]",
                    conn.execute(f"PRAGMA table_info({table})").fetchall(),
                )
                assert sum(row[5] for row in keys) == 1
                foreign = cast(
                    "list[tuple[object, object, str, str]]",
                    conn.execute(f"PRAGMA foreign_key_list({table})").fetchall(),
                )
                assert any(
                    row[2] == "coordinator_sessions" and row[3] == "session_id" for row in foreign
                )
        graph.close()
