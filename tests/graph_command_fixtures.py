from typing import cast

from milknado.domains.common import SessionAction
from milknado.domains.graph import GraphCommand, MikadoGraph

NOW = "2026-09-12T12:00:00+00:00"
LATER = "2026-09-12T12:05:00+00:00"


def owned_graph(
    graph: MikadoGraph, *, owner: str = "owner-1", invocation: str = "invoke-1"
) -> int:
    node = graph.add_node("steerable")
    assert graph.claim_node(node.id, "run-1", now=NOW)
    graph.runs.start("run-1", node.id, "run.log", NOW, 60)
    _ = graph.commands.publish_capabilities(
        "run-1",
        node.id,
        invocation,
        owner,
        ("steer", "approve", "deny"),
        ("permission-1",),
        published_at=NOW,
    )
    return node.id


def command(  # noqa: PLR0913
    node_id: int,
    *,
    command_id: str = "command-1",
    owner: str = "owner-1",
    invocation: str = "invoke-1",
    action: str = "steer",
    text: str = "redirect",
    permission_id: str | None = None,
    expires_at: str = LATER,
) -> GraphCommand:
    return GraphCommand(
        command_id=command_id,
        node_id=node_id,
        run_id="run-1",
        invocation_id=invocation,
        owner_incarnation=owner,
        action=cast(SessionAction, action),
        text=text,
        permission_id=permission_id,
        expires_at=expires_at,
    )
