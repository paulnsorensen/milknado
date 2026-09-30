# pyright: reportAny=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownArgumentType=false
import pytest

from milknado.web import HostDependencies, WebCommands, observer_commands, owner_commands
from tests.web.support import client, headers


def _commands(graph):
    node = graph.add_node("steerable")
    assert graph.claim_node(node.id, "run-1", now="2026-09-12T12:00:00+00:00")
    graph.runs.start("run-1", node.id, "run.log", "2026-09-12T12:00:00+00:00", 60)
    graph.commands.publish_capabilities(
        "run-1", node.id, "inv-1", "owner-1", ("steer",), published_at="2026-09-12T12:00:00+00:00"
    )
    return observer_commands(dependencies=HostDependencies(graph=graph))


def test_session_input_is_admitted_once(graph) -> None:
    commands = _commands(graph)
    test_client = client(commands)[0]
    payload = {"command_id": "cmd-1", "action": "steer", "text": "hello"}
    first = test_client.post("/api/runs/run-1/session-input", json=payload, headers=headers())
    second = test_client.post("/api/runs/run-1/session-input", json=payload, headers=headers())
    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json() == second.json()
    assert graph.commands.command("cmd-1") is not None


def test_session_input_rejects_conflicting_command_id(graph) -> None:
    commands = _commands(graph)
    test_client = client(commands)[0]
    first = test_client.post(
        "/api/runs/run-1/session-input",
        json={"command_id": "cmd-1", "action": "steer", "text": "hello"},
        headers=headers(),
    )
    conflict = test_client.post(
        "/api/runs/run-1/session-input",
        json={"command_id": "cmd-1", "action": "steer", "text": "different"},
        headers=headers(),
    )
    assert first.status_code == 200
    assert conflict.status_code == 409
    assert "different command" in conflict.json()["reason"]


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        ({"command_id": "  ", "action": "steer", "text": "hello"}, "command_id"),
        ({"command_id": "cmd-2", "action": "steer"}, "text"),
        ({"command_id": "cmd-2b", "action": "follow_up"}, "text"),
        ({"command_id": "cmd-3", "action": "approve"}, "request_id"),
    ],
)
def test_session_input_rejects_malformed_payload(graph, payload, reason) -> None:
    commands = _commands(graph)
    response = client(commands)[0].post(
        "/api/runs/run-1/session-input", json=payload, headers=headers()
    )
    assert response.status_code == 400
    assert reason in response.json()["reason"]


def test_session_input_rejection_surfaces_handler_reason() -> None:
    def reject(run_id: str, request: object) -> None:
        del run_id, request
        raise ValueError("the run is not active")

    commands = WebCommands(session_input=reject)
    response = client(commands)[0].post(
        "/api/runs/run-1/session-input",
        json={"command_id": "cmd-rejected", "action": "interrupt"},
        headers=headers(),
    )

    assert response.status_code == 409
    assert response.json()["reason"] == "Session input was rejected: the run is not active."


def test_session_input_rejection_falls_back_to_unavailable_when_handler_returns_none() -> None:
    commands = WebCommands(session_input=lambda run_id, request: None)
    response = client(commands)[0].post(
        "/api/runs/run-1/session-input",
        json={"command_id": "cmd-rejected", "action": "interrupt"},
        headers=headers(),
    )

    assert response.status_code == 409
    assert response.json()["reason"] == "Session input was rejected: the command is unavailable."


class _DecliningController:
    def session_input(self, run_id, command) -> bool:
        del run_id, command
        return False

    def cancel(self, run_id) -> dict[str, object] | None:
        del run_id
        return None

    def force_stop(self, run_id, timeout: float = 10.0) -> bool:
        del run_id, timeout
        return False

    def stop_scheduling(self) -> None:
        return None


def test_owner_session_input_reports_unavailable_when_controller_declines() -> None:
    commands = owner_commands(_DecliningController())
    response = client(commands)[0].post(
        "/api/runs/run-1/session-input",
        json={"command_id": "cmd-declined", "action": "interrupt"},
        headers=headers(),
    )

    assert response.status_code == 409
    assert response.json()["reason"] == "Session input was rejected: the command is unavailable."


def test_session_input_rejection_reports_run_not_active_for_missing_run(graph) -> None:
    commands = observer_commands(dependencies=HostDependencies(graph=graph))
    response = client(commands)[0].post(
        "/api/runs/run-1/session-input",
        json={"command_id": "cmd-rejected", "action": "interrupt"},
        headers=headers(),
    )

    assert response.status_code == 409
    assert response.json()["reason"] == "Session input was rejected: the run is not active."


def test_session_input_rejection_reports_unavailable_for_disallowed_action(graph) -> None:
    commands = _commands(graph)
    response = client(commands)[0].post(
        "/api/runs/run-1/session-input",
        json={"command_id": "cmd-rejected", "action": "follow_up", "text": "hi"},
        headers=headers(),
    )

    assert response.status_code == 409
    assert response.json()["reason"] == "Session input was rejected: the command is unavailable."


def test_session_input_admits_interrupt_without_text(graph) -> None:
    node = graph.add_node("interruptible")
    assert graph.claim_node(node.id, "run-1", now="2026-09-12T12:00:00+00:00")
    graph.runs.start("run-1", node.id, "run.log", "2026-09-12T12:00:00+00:00", 60)
    graph.commands.publish_capabilities(
        "run-1",
        node.id,
        "inv-1",
        "owner-1",
        ("interrupt",),
        published_at="2026-09-12T12:00:00+00:00",
    )
    commands = observer_commands(dependencies=HostDependencies(graph=graph))
    response = client(commands)[0].post(
        "/api/runs/run-1/session-input",
        json={"command_id": "cmd-4", "action": "interrupt"},
        headers=headers(),
    )
    assert response.status_code == 200
    assert graph.commands.command("cmd-4") is not None
