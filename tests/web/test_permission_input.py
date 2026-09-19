# pyright: reportAny=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownArgumentType=false
import pytest

from milknado.web import HostDependencies, observer_commands
from tests.web.support import client, headers


@pytest.mark.parametrize("decision", ("approve", "deny"))
def test_permission_input_uses_owner_capability_fence(graph, decision) -> None:
    node = graph.add_node("permission")
    assert graph.claim_node(node.id, "run-1", now="2026-09-12T12:00:00+00:00")
    graph.runs.start("run-1", node.id, "run.log", "2026-09-12T12:00:00+00:00", 60)
    graph.commands.publish_capabilities(
        "run-1",
        node.id,
        "inv-1",
        "owner-1",
        (decision,),
        ("permission-1",),
        published_at="2026-09-12T12:00:00+00:00",
    )
    commands = observer_commands(dependencies=HostDependencies(graph=graph))
    response = client(commands)[0].post(
        "/api/runs/run-1/session-input",
        json={"command_id": f"cmd-{decision}", "action": decision, "request_id": "permission-1"},
        headers=headers(),
    )
    assert response.status_code == 200
    assert response.json()["action"] == decision
    assert response.json()["request_id"] == "permission-1"
    stored = graph.commands.command(f"cmd-{decision}")
    assert stored is not None
    assert stored.owner_incarnation == "owner-1"
    assert stored.invocation_id == "inv-1"
    assert stored.action == decision
    assert stored.permission_id == "permission-1"
