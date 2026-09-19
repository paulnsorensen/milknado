from typing import cast
from unittest.mock import Mock

import pytest

from milknado.domains.common import SessionInput
from milknado.domains.graph import MikadoGraph
from milknado.domains.graph.commands import OwnerCapabilities
from milknado.web import (
    HostDependencies,
    ObserverHandlers,
    OwnerHandlers,
    WebCommands,
    observer_commands,
    owner_commands,
)
from tests.web.support import client, headers


def _snapshot_capabilities(commands: WebCommands) -> dict[str, dict[str, object]]:
    response = client(commands)[0].get("/api/snapshot", headers=headers())  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    return cast(
        dict[str, dict[str, object]],
        response.json()["capabilities"],  # pyright: ignore[reportUnknownMemberType]
    )


def test_owner_builder_exposes_owner_capability_matrix() -> None:
    owner = OwnerCapabilities(
        run_id="run-1",
        node_id=1,
        invocation_id="inv-1",
        owner_incarnation="owner-1",
        actions=(),
        permission_ids=(),
        published_at="now",
    )
    commands = owner_commands(
        OwnerHandlers(
            session_input=lambda run_id, request: None,
            cancel=lambda run_id: {"run_id": run_id},
            force_stop=lambda run_id: {"run_id": run_id},
            stop_scheduling=lambda: None,
        ),
        dependencies=HostDependencies(owner_capabilities=owner),
    )
    capabilities = _snapshot_capabilities(commands)
    assert all(
        capabilities[name]["available"]
        for name in ("session_input", "cancel", "force_stop", "stop_scheduling")
    )
    assert capabilities["owner"]["available"] is True
    assert capabilities["owner"]["published_at"] == "now"


def test_snapshot_reads_live_owner_capabilities_after_app_construction() -> None:
    first = OwnerCapabilities(
        run_id="run-1",
        node_id=1,
        invocation_id="inv-1",
        owner_incarnation="owner-1",
        actions=(),
        permission_ids=(),
        published_at="first",
    )
    second = OwnerCapabilities(
        run_id="run-2",
        node_id=2,
        invocation_id="inv-2",
        owner_incarnation="owner-2",
        actions=("steer",),
        permission_ids=("permission-2",),
        published_at="second",
    )
    current = [first]
    commands = WebCommands(owner_capabilities=lambda: current[0])
    test_client, _ = client(commands)

    first_response = test_client.get("/api/snapshot", headers=headers())  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    current[0] = second
    second_response = test_client.get("/api/snapshot", headers=headers())  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]

    assert first_response.json()["capabilities"]["owner"]["published_at"] == "first"  # pyright: ignore[reportUnknownMemberType]
    assert second_response.json()["capabilities"]["owner"]["run_id"] == "run-2"  # pyright: ignore[reportUnknownMemberType]
    assert second_response.json()["capabilities"]["owner"]["actions"] == ["steer"]  # pyright: ignore[reportUnknownMemberType]
    assert second_response.json()["capabilities"]["owner"]["permission_ids"] == ["permission-2"]  # pyright: ignore[reportUnknownMemberType]


def test_observer_session_input_reads_new_owner_fence_per_request(
    graph: MikadoGraph,
) -> None:
    node = graph.add_node("steerable")
    assert graph.claim_node(node.id, "run-1", now="2026-09-12T12:00:00+00:00")
    graph.runs.start("run-1", node.id, "run.log", "2026-09-12T12:00:00+00:00", 60)
    first = OwnerCapabilities(
        run_id="run-1",
        node_id=node.id,
        invocation_id="inv-1",
        owner_incarnation="owner-1",
        actions=("steer",),
        permission_ids=(),
        published_at="first",
    )
    second = OwnerCapabilities(
        run_id="run-1",
        node_id=node.id,
        invocation_id="inv-2",
        owner_incarnation="owner-2",
        actions=("steer",),
        permission_ids=(),
        published_at="second",
    )
    current = [first]
    commands = observer_commands(
        dependencies=HostDependencies(graph=graph, owner_capabilities=lambda: current[0])
    )
    assert commands.session_input is not None
    current[0] = second
    _ = graph.commands.publish_capabilities(
        "run-1",
        node.id,
        "inv-2",
        "owner-2",
        ("steer",),
        (),
        published_at="2026-09-12T12:00:01+00:00",
    )
    command = SessionInput(action="steer", request_id="request-1")

    admitted = commands.session_input("run-1", command)

    assert admitted is not None
    assert admitted.command_id == "request-1"


def test_observer_builder_reports_owner_only_commands_unavailable() -> None:
    owner = OwnerCapabilities(
        run_id="run-1",
        node_id=1,
        invocation_id="inv-1",
        owner_incarnation="owner-1",
        actions=(),
        permission_ids=(),
        published_at="now",
    )
    capabilities = _snapshot_capabilities(
        observer_commands(ObserverHandlers(), HostDependencies(owner_capabilities=owner))
    )
    assert capabilities["session_input"]["available"] is False
    assert capabilities["cancel"]["available"] is False
    assert capabilities["force_stop"]["available"] is False
    assert capabilities["stop_scheduling"]["available"] is False
    assert capabilities["graph_edits"]["available"] is False
    assert capabilities["review_decision"]["available"] is False
    assert capabilities["git"]["available"] is False
    assert capabilities["owner"]["available"] is True
    assert capabilities["owner"]["run_id"] == "run-1"
    assert capabilities["owner"]["published_at"] == "now"


class _Controller:
    def __init__(self) -> None:
        self.run_ids: list[str] = []

    def session_input(self, run_id: str, command: SessionInput) -> bool:
        self.run_ids.append(run_id)
        _ = command
        return True

    def cancel(self, run_id: str) -> dict[str, object]:
        self.run_ids.append(run_id)
        return {"run_id": run_id, "state": "cancelled"}

    def force_stop(self, run_id: str, timeout: float = 10.0) -> bool:
        self.run_ids.append(run_id)
        _ = timeout
        return True

    def stop_scheduling(self) -> None:
        pass


def test_owner_controller_builder_targets_requested_run_id() -> None:
    controller = _Controller()
    owner = OwnerCapabilities(
        run_id="owner-run",
        node_id=1,
        invocation_id="inv-1",
        owner_incarnation="owner-1",
        actions=(),
        permission_ids=(),
        published_at="now",
    )
    commands = owner_commands(controller, HostDependencies(owner_capabilities=owner))
    assert commands.session_input is not None
    command = SessionInput(action="steer", request_id="request")
    assert commands.session_input("request-run", command) is None
    assert commands.session_input("owner-run", command) == command
    assert commands.cancel is not None
    assert commands.cancel("owner-run") == {"run_id": "owner-run", "state": "cancelled"}
    assert controller.run_ids == ["owner-run", "owner-run"]


def test_observer_builder_wires_graph_process_and_project_dependencies(
    graph: MikadoGraph,
) -> None:
    commands = observer_commands(
        dependencies=HostDependencies(
            graph=graph,
            git_port=Mock(),
            process=Mock(),
        )
    )
    assert commands.session_input is not None
    assert (
        commands.session_input("missing-run", SessionInput(action="steer", request_id="request"))
        is None
    )
    assert commands.cancel is not None
    with pytest.raises(ValueError, match="not found"):
        _ = commands.cancel("missing-run")
