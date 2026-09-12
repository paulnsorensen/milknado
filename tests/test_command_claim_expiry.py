from __future__ import annotations

from typing import cast

from milknado.domains.common import SessionAction
from milknado.domains.graph import GraphCommand, MikadoGraph

_NOW = "2026-09-12T12:00:00+00:00"
_LATER = "2026-09-12T12:05:00+00:00"


def test_claim_expires_current_invocation_and_records_receipt(graph: MikadoGraph) -> None:
    node = graph.add_node("steerable")
    assert graph.claim_node(node.id, "run-1", now=_NOW)
    graph.runs.start("run-1", node.id, "run.log", _NOW, 60)
    _ = graph.commands.publish_capabilities(
        "run-1",
        node.id,
        "invoke-1",
        "owner-1",
        ("steer",),
        published_at=_NOW,
    )
    command = GraphCommand(
        command_id="command-expiring",
        node_id=node.id,
        run_id="run-1",
        invocation_id="invoke-1",
        owner_incarnation="owner-1",
        action=cast(SessionAction, "steer"),
        text="redirect",
        expires_at=_LATER,
    )

    assert graph.commands.admit(command, now=_NOW).status == "queued"
    assert graph.commands.claim_pending("run-1", "owner-1", now=_LATER) == ()
    assert [receipt.status for receipt in graph.commands.history(command.command_id)] == [
        "queued",
        "expired",
    ]


def test_claim_expires_previous_invocation_and_records_receipt(graph: MikadoGraph) -> None:
    node = graph.add_node("steerable")
    assert graph.claim_node(node.id, "run-1", now=_NOW)
    graph.runs.start("run-1", node.id, "run.log", _NOW, 60)
    _ = graph.commands.publish_capabilities(
        "run-1", node.id, "invoke-1", "owner-1", ("steer",), published_at=_NOW
    )
    command = GraphCommand(
        command_id="command-stale",
        node_id=node.id,
        run_id="run-1",
        invocation_id="invoke-1",
        owner_incarnation="owner-1",
        action=cast(SessionAction, "steer"),
        text="redirect",
        expires_at=_LATER,
    )
    assert graph.commands.admit(command, now=_NOW).status == "queued"
    _ = graph.commands.publish_capabilities(
        "run-1", node.id, "invoke-2", "owner-1", ("steer",), published_at=_LATER
    )

    assert graph.commands.claim_pending("run-1", "owner-1", now=_LATER) == ()
    assert [receipt.status for receipt in graph.commands.history(command.command_id)] == [
        "queued",
        "expired",
    ]
