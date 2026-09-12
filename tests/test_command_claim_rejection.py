from __future__ import annotations

import pytest

from milknado.domains.graph import MikadoGraph
from tests.graph_command_fixtures import NOW as _NOW
from tests.graph_command_fixtures import command as _command
from tests.graph_command_fixtures import owned_graph as _owned_graph


@pytest.mark.parametrize(
    ("actions", "permission_ids", "detail"),
    [
        (
            ("approve",),
            (),
            "permission ID does not exactly match a current permission",
        ),
        (("approve",), ("permission-1",), "action 'steer' is not currently available"),
    ],
)
def test_claim_rejects_current_owner_command_after_capability_removal(
    graph: MikadoGraph,
    actions: tuple[str, ...],
    permission_ids: tuple[str, ...],
    detail: str,
) -> None:
    node_id = _owned_graph(graph)
    action = "approve" if permission_ids == () else "steer"
    permission_id = "permission-1" if permission_ids == () else None
    command = _command(node_id, action=action, permission_id=permission_id)
    assert graph.commands.admit(command, now=_NOW).status == "queued"
    _ = graph.commands.publish_capabilities(
        "run-1", node_id, "invoke-1", "owner-1", actions, permission_ids, published_at=_NOW
    )

    assert graph.commands.claim_pending("run-1", "owner-1", now=_NOW) == ()
    stored = graph.commands.command(command.command_id)
    assert stored is not None
    assert stored.status == "rejected"
    receipt = graph.commands.receipt(command.command_id)
    assert receipt is not None
    assert receipt.detail == detail
    assert [receipt.status for receipt in graph.commands.history(command.command_id)] == [
        "queued",
        "rejected",
    ]


def test_stale_owner_claim_does_not_reject_replacement_owner_command(
    graph: MikadoGraph,
) -> None:
    node_id = _owned_graph(graph)
    replacement = _command(
        node_id,
        command_id="replacement-command",
        owner="owner-2",
        invocation="invoke-2",
        action="approve",
        permission_id="permission-1",
    )
    _ = graph.commands.publish_capabilities(
        "run-1", node_id, "invoke-2", "owner-2", ("approve",), ("permission-1",), published_at=_NOW
    )
    assert graph.commands.admit(replacement, now=_NOW).status == "queued"

    assert graph.commands.claim_pending("run-1", "owner-1", now=_NOW) == ()
    stored = graph.commands.command(replacement.command_id)
    assert stored is not None
    assert stored.status == "queued"
    assert [receipt.status for receipt in graph.commands.history(replacement.command_id)] == [
        "queued"
    ]
    claimed = graph.commands.claim_pending("run-1", "owner-2", now=_NOW)
    assert len(claimed) == 1
    assert claimed[0].command_id == replacement.command_id
