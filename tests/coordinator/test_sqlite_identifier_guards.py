from pathlib import Path
from typing import cast

import msgspec
import pytest

from milknado.domains.coordinator import (
    CoordinatorCommand,
    CoordinatorCommandReceipt,
    CoordinatorControl,
    StartGoal,
)
from milknado.domains.coordinator.control_models import CreateGroup, RecordRevision
from milknado.domains.graph import MikadoGraph


def _payload(kind: str, node_id: int) -> dict[str, object]:
    if kind == "create_group":
        return {
            "kind": kind,
            "command_id": "group",
            "graph_id": "main",
            "tasks": [node_id],
            "worktree_path": "/tmp/group",
            "branch_name": "branch",
            "provider_session_id": "provider",
        }
    return {
        "kind": kind,
        "command_id": "revision",
        "revision_id": "rev-1",
        "affected_node_ids": [node_id],
    }


def _send_json(
    control: CoordinatorControl, session_id: str, payload: dict[str, object]
) -> CoordinatorCommandReceipt:
    encoded = msgspec.json.encode(payload)
    command = cast(CoordinatorCommand, msgspec.json.decode(encoded, type=CoordinatorCommand))
    return control.send_coordinator_command(session_id, command)


@pytest.mark.parametrize("kind", ["create_group", "record_revision"])
@pytest.mark.parametrize("node_id", [-(2**63) - 1, 2**63])
def test_out_of_range_decoded_identifier_leaves_receipts_unchanged(
    tmp_path: Path, kind: str, node_id: int
) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    control = CoordinatorControl(graph, tmp_path)
    started = control.send_coordinator_command("", StartGoal("start", "Deliver", "codex"))
    session_id = cast(str, cast(dict[str, object], started.result)["id"])
    before = graph.group_connection.execute(
        "SELECT command_id, status FROM coordinator_web_receipts ORDER BY command_id"
    ).fetchall()

    with pytest.raises(msgspec.ValidationError):
        _ = _send_json(control, session_id, _payload(kind, node_id))

    after = graph.group_connection.execute(
        "SELECT command_id, status FROM coordinator_web_receipts ORDER BY command_id"
    ).fetchall()
    assert after == before
    graph.close()


@pytest.mark.parametrize("kind", ["create_group", "record_revision"])
@pytest.mark.parametrize("node_id", [-(2**63), 2**63 - 1])
def test_signed_64_bit_limits_keep_json_shape(kind: str, node_id: int) -> None:
    payload = _payload(kind, node_id)
    encoded = msgspec.json.encode(payload)
    command = cast(CoordinatorCommand, msgspec.json.decode(encoded, type=CoordinatorCommand))
    assert isinstance(command, CreateGroup if kind == "create_group" else RecordRevision)
    assert (command.tasks if isinstance(command, CreateGroup) else command.affected_node_ids) == (
        node_id,
    )
    assert msgspec.json.decode(msgspec.json.encode(command)) == payload


def test_decoded_commands_keep_effects_and_replay(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    control = CoordinatorControl(graph, tmp_path)
    started = control.send_coordinator_command("", StartGoal("start", "Deliver", "codex"))
    result = cast(dict[str, object], started.result)
    session_id = cast(str, result["id"])
    task = graph.add_node("Task", cast(int, result["goal_id"]))

    revision_payload = _payload("record_revision", task.id)
    revision = _send_json(control, session_id, revision_payload)
    assert revision.status == "accepted"
    assert _send_json(control, session_id, revision_payload) == revision
    assert any(
        event.kind == "graph_revision"
        for event in control.read_coordinator_snapshot(session_id, 0).events
    )

    group_payload = _payload("create_group", task.id)
    group = _send_json(control, session_id, group_payload)
    assert group.status == "accepted"
    assert _send_json(control, session_id, group_payload) == group
    assert len(control.read_coordinator_snapshot(session_id, 0).groups) == 1
    graph.close()
