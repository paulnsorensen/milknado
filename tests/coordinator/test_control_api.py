from pathlib import Path
from typing import cast

import pytest

from milknado.domains.common import SessionInput
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
    snapshot = control.read_coordinator_snapshot(session_id, 0)
    assert [event.seq for event in snapshot.events] == sorted(
        event.seq for event in snapshot.events
    )
    assert {link.kind for link in snapshot.links} >= {"execution_group", "run", "provider_session"}
    assert snapshot.groups[0].id == group_id
    assert snapshot.cursor == snapshot.events[-1].seq
    assert control.read_coordinator_snapshot(session_id, snapshot.cursor).events == ()
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
    count = graph.group_connection.execute("SELECT COUNT(*) FROM coordinator_sessions").fetchone()
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
