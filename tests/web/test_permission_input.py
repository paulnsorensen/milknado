# pyright: reportAny=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownArgumentType=false
from milknado.web import HostDependencies, observer_commands
from tests.web.support import client, headers


def test_permission_input_uses_owner_capability_fence(graph) -> None:
    node = graph.add_node("permission")
    assert graph.claim_node(node.id, "run-1", now="2026-09-12T12:00:00+00:00")
    graph.runs.start("run-1", node.id, "run.log", "2026-09-12T12:00:00+00:00", 60)
    graph.commands.publish_capabilities(
        "run-1",
        node.id,
        "inv-1",
        "owner-1",
        ("approve",),
        ("permission-1",),
        published_at="2026-09-12T12:00:00+00:00",
    )
    commands = observer_commands(dependencies=HostDependencies(graph=graph))
    response = client(commands)[0].post(
        "/api/runs/run-1/session-input",
        json={"command_id": "cmd-approve", "action": "approve", "request_id": "permission-1"},
        headers=headers(),
    )
    assert response.status_code == 200
    assert response.json()["action"] == "approve"
    assert response.json()["request_id"] == "permission-1"
    assert graph.commands.command("cmd-approve") is not None
