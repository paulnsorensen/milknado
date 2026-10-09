from pathlib import Path
from typing import cast

import msgspec
import pytest

from milknado.domains.coordinator import CoordinatorControl
from milknado.domains.coordinator.control_models import StartGoal
from milknado.domains.coordinator.projection import ProviderTurnState
from milknado.domains.coordinator.receipt_results import (
    PlanCommandResult,
    RecoveryCommandResult,
    RecoveryItem,
    UnknownTurnItem,
)
from milknado.domains.graph import MikadoGraph


def test_receipt_and_snapshot_are_frozen_boundary_schemas(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    control = CoordinatorControl(graph, tmp_path)
    command = StartGoal("start", "Deliver", "codex")
    receipt = control.send_coordinator_command("", command)
    session_id = cast(str, cast(dict[str, object], receipt.result)["id"])
    snapshot = control.read_coordinator_snapshot(session_id, 0)

    for value in (receipt, snapshot, ProviderTurnState("codex", "provider", "turn", "done")):
        assert isinstance(value, msgspec.Struct)
        assert type(value).__struct_config__.frozen
        with pytest.raises(AttributeError):
            setattr(value, msgspec.structs.fields(type(value))[0].name, None)

    assert msgspec.json.decode(msgspec.json.encode(receipt))["command_id"] == "start"
    payload = msgspec.json.decode(msgspec.json.encode(snapshot), type=dict[str, object])
    assert cast(dict[str, object], payload["session"])["id"] == session_id
    assert cast(dict[str, object], payload["goal"])["description"] == "Deliver"
    assert payload["provider_turns"] == []
    graph.close()


def test_nested_receipt_results_keep_json_fields() -> None:
    item = RecoveryItem("run", "run-1", "codex", "provider", "/tmp/worktree", "ready")
    unknown = UnknownTurnItem("codex", "provider", "turn")
    recovery = RecoveryCommandResult("session", (item,), (unknown,))
    plan = PlanCommandResult(True, 0, None, 1, 2, 0, "optimal", 3, None)

    for value in (plan, item, unknown, recovery):
        assert isinstance(value, msgspec.Struct)
        assert type(value).__struct_config__.frozen
        with pytest.raises(AttributeError):
            setattr(value, msgspec.structs.fields(type(value))[0].name, None)

    assert msgspec.json.decode(msgspec.json.encode(plan))["nodes_created"] == 1
    assert msgspec.json.decode(msgspec.json.encode(recovery)) == {
        "session_id": "session",
        "receipts": [
            {
                "entity_kind": "run",
                "entity_id": "run-1",
                "provider_family": "codex",
                "provider_session_id": "provider",
                "worktree_path": "/tmp/worktree",
                "outcome": "ready",
            }
        ],
        "unknown_turns": [
            {
                "provider_family": "codex",
                "provider_session_id": "provider",
                "turn_id": "turn",
            }
        ],
    }
