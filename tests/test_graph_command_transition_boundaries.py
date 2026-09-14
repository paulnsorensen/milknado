from __future__ import annotations

from typing import cast

import pytest

from milknado.domains.graph import CommandFenceError, CommandStatus, MikadoGraph
from tests.graph_command_fixtures import NOW, command, owned_graph

EXACT_EXPIRY = "2026-09-12T12:01:00+00:00"


def test_public_limits_reject_invalid_pending_and_claim_bounds(
    graph: MikadoGraph,
) -> None:
    node_id = owned_graph(graph)
    value = command(node_id)
    with pytest.raises(ValueError, match="positive"):
        _ = graph.commands.admit(value, now=NOW, max_pending=0)
    assert graph.commands.history(value.command_id) == ()
    with pytest.raises(ValueError, match="between"):
        _ = graph.commands.pending("run-1", limit=0)
    with pytest.raises(ValueError, match="between"):
        _ = graph.commands.pending("run-1", limit=65)
    with pytest.raises(ValueError, match="between"):
        _ = graph.commands.claim_pending("run-1", "owner-1", limit=0)


def test_duplicate_submitted_and_delivered_receipts_are_idempotent(
    graph: MikadoGraph,
) -> None:
    node_id = owned_graph(graph)
    value = command(node_id)
    assert graph.commands.admit(value, now=NOW).status == "queued"
    submitted = graph.commands.submit(value, now=NOW)
    assert graph.commands.submit(value, now=NOW) == submitted
    delivered = graph.commands.deliver(value, now=NOW)
    assert graph.commands.deliver(value, now=NOW) == delivered
    assert [receipt.status for receipt in graph.commands.history(value.command_id)] == [
        "queued",
        "submitted",
        "delivered",
    ]


def test_unknown_transition_and_expiry_use_exact_public_boundaries(
    graph: MikadoGraph,
) -> None:
    node_id = owned_graph(graph)
    with pytest.raises(ValueError, match="unknown command"):
        _ = graph.commands.transition(
            "missing",
            "submitted",
            node_id=node_id,
            run_id="run-1",
            invocation_id="invoke-1",
            owner_incarnation="owner-1",
            now=NOW,
        )
    value = command(node_id, expires_at=EXACT_EXPIRY)
    assert graph.commands.admit(value, now=NOW).status == "queued"
    expired = graph.commands.submit(value, now=EXACT_EXPIRY)
    assert expired.status == "expired"
    assert [item.status for item in graph.commands.history(value.command_id)] == [
        "queued",
        "expired",
    ]


def test_stale_fence_rejects_transition_without_history_change(
    graph: MikadoGraph,
) -> None:
    node_id = owned_graph(graph)
    value = command(node_id)
    assert graph.commands.admit(value, now=NOW).status == "queued"
    history = graph.commands.history(value.command_id)
    with pytest.raises(CommandFenceError, match="fence"):
        _ = graph.commands.transition(
            value.command_id,
            "submitted",
            node_id=node_id,
            run_id="run-1",
            invocation_id="invoke-1",
            owner_incarnation="old-owner",
            now=NOW,
        )
    assert graph.commands.history(value.command_id) == history


def test_current_capability_fence_and_permission_reject_transition(
    graph: MikadoGraph,
) -> None:
    node_id = owned_graph(graph)
    value = command(node_id)
    assert graph.commands.admit(value, now=NOW).status == "queued"
    history = graph.commands.history(value.command_id)
    _ = graph.commands.publish_capabilities(
        "run-1", node_id, "invoke-2", "owner-2", ("steer",), published_at=NOW
    )
    with pytest.raises(ValueError, match="fence"):
        _ = graph.commands.submit(value, now=NOW)
    assert graph.commands.history(value.command_id) == history

    permission = command(
        node_id, command_id="permission-command", action="approve", permission_id="permission-1"
    )
    _ = graph.commands.publish_capabilities(
        "run-1",
        node_id,
        "invoke-1",
        "owner-1",
        ("approve",),
        ("permission-2",),
        published_at=NOW,
    )
    receipt = graph.commands.admit(permission, now=NOW)
    assert receipt.status == "rejected"
    assert receipt.detail == "permission ID does not exactly match a current permission"
    assert graph.commands.history(permission.command_id) == (receipt,)


def test_invalid_and_skipped_transitions_reject(graph: MikadoGraph) -> None:
    node_id = owned_graph(graph)
    value = command(node_id)
    assert graph.commands.admit(value, now=NOW).status == "queued"
    with pytest.raises(ValueError, match="invalid command transition"):
        _ = graph.commands.transition(
            value.command_id,
            "delivered",
            node_id=node_id,
            run_id="run-1",
            invocation_id="invoke-1",
            owner_incarnation="owner-1",
            now=NOW,
        )
    with pytest.raises(ValueError, match="invalid command transition"):
        _ = graph.commands.transition(
            value.command_id,
            cast(CommandStatus, cast(object, "bogus")),
            node_id=node_id,
            run_id="run-1",
            invocation_id="invoke-1",
            owner_incarnation="owner-1",
            now=NOW,
        )
    assert [item.status for item in graph.commands.history(value.command_id)] == ["queued"]
