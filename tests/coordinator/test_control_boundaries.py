from pathlib import Path
from typing import cast

import pytest

from milknado.domains.common import SessionContext, SessionEvent, SessionInput
from milknado.domains.coordinator import CoordinatorControl, CoordinatorServices, ProviderBinding
from milknado.domains.coordinator.control_models import (
    CreateGroup,
    DispatchTask,
    FailLaunch,
    RecordRevision,
    RuntimeAction,
    StartGoal,
)
from milknado.domains.coordinator.persistence import bind_provider_session
from milknado.domains.graph import MikadoGraph
from milknado.loop.sessions import ProviderSessionIdentity, RuntimeSession, SessionChannel


def _result(value: object) -> dict[str, object]:
    return cast(dict[str, object], value)


def test_revision_and_launch_failure_replay_without_duplicate_events(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    control = CoordinatorControl(graph, tmp_path)
    started = control.send_coordinator_command("", StartGoal("start", "Deliver", "codex"))
    session_id = cast(str, _result(started.result)["id"])
    task = graph.add_node("Task", cast(int, _result(started.result)["goal_id"]))
    revision = RecordRevision("revision", "rev-1", (task.id,))
    first = control.send_coordinator_command(session_id, revision)
    assert first.status == "accepted" and first.result is None
    assert control.send_coordinator_command(session_id, revision) == first

    group = control.send_coordinator_command(
        session_id,
        CreateGroup("group", "main", (task.id,), str(tmp_path / "group"), "branch", "provider"),
    )
    group_id = cast(str, _result(group.result)["id"])
    dispatched = control.send_coordinator_command(
        session_id, DispatchTask("dispatch", group_id, task.id, "run")
    )
    attempt = cast(str, _result(_result(dispatched.result)["attempt"])["attempt_id"])
    failure = FailLaunch("fail", group_id, task.id, "run", attempt, "provider failed")
    failed = control.send_coordinator_command(session_id, failure)
    assert failed.status == "accepted" and failed.result is None
    assert graph.groups.task_result(task.id) == ("failed", "provider failed")
    before = control.read_coordinator_snapshot(session_id, 0).events
    assert control.send_coordinator_command(session_id, failure) == failed
    assert control.read_coordinator_snapshot(session_id, 0).events == before
    graph.close()


def test_connected_runtime_action_queues_once_and_replays(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    channel = SessionChannel()
    channel.start(SessionContext(family="codex", cwd=str(tmp_path)), ("approve",))
    channel.publish(
        SessionEvent(kind="permission", text="write?", event_id="p-1", state="requested")
    )
    incarnation = channel.capture_incarnation()
    assert incarnation is not None
    runtime = RuntimeSession(ProviderSessionIdentity("codex", "provider"), channel, incarnation)
    control = CoordinatorControl(
        graph,
        tmp_path,
        CoordinatorServices(
            runtime_session=lambda identity: runtime if identity == "provider" else None
        ),
    )
    started = control.send_coordinator_command("", StartGoal("start", "Deliver", "codex"))
    session_id = cast(str, _result(started.result)["id"])
    task = graph.add_node("Task", cast(int, _result(started.result)["goal_id"]))
    group = control.send_coordinator_command(
        session_id,
        CreateGroup("group", "main", (task.id,), str(tmp_path / "group"), "branch", "provider"),
    )
    assert group.status == "accepted"
    bind_provider_session(
        graph.group_connection,
        session_id,
        ProviderBinding(
            "execution_group", cast(str, _result(group.result)["id"]), "codex", "provider"
        ),
    )
    action = RuntimeAction(
        "action",
        "provider",
        SessionInput(action="approve", request_id=channel.view().permissions[0].event_id),
    )
    first = control.send_coordinator_command(session_id, action)
    assert first.status == "accepted"
    assert _result(first.result)["state"] == "queued"
    assert control.send_coordinator_command(session_id, action) == first
    assert len(channel.drain()) == 1
    state = cast(
        tuple[str] | None,
        graph.group_connection.execute(
            "SELECT state FROM coordinator_action_receipts WHERE command_id = 'action'"
        ).fetchone(),
    )
    assert state is not None and state[0] == "queued"
    graph.close()


def test_empty_command_identity_leaves_store_unchanged(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    control = CoordinatorControl(graph, tmp_path)
    with pytest.raises(ValueError, match="command_id"):
        _ = control.send_coordinator_command("", StartGoal("", "Deliver", "codex"))
    receipts = cast(
        tuple[int] | None,
        graph.group_connection.execute("SELECT COUNT(*) FROM coordinator_web_receipts").fetchone(),
    )
    assert receipts is not None and receipts[0] == 0
    graph.close()
