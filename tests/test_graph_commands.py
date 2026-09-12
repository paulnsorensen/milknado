from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import cast

import pytest

from milknado.app.watch import graph_command_admitter
from milknado.domains.common import SessionInput
from milknado.domains.graph import (
    CommandFenceError,
    MikadoGraph,
    new_command_id,
)
from tests.graph_command_fixtures import (
    LATER as _LATER,
)
from tests.graph_command_fixtures import (
    NOW as _NOW,
)
from tests.graph_command_fixtures import (
    command as _command,
)
from tests.graph_command_fixtures import (
    owned_graph as _owned_graph,
)
from tests.graph_helpers import graph_conn


def test_command_schema_is_separate_and_owner_capabilities_are_explicit(
    tmp_path: Path,
) -> None:
    graph = MikadoGraph(tmp_path / "commands.db")
    node_id = _owned_graph(graph)

    capabilities = graph.commands.capabilities("run-1")
    assert capabilities is not None
    assert capabilities.owner_incarnation == "owner-1"
    assert capabilities.actions == ("steer", "approve", "deny")
    assert capabilities.permission_ids == ("permission-1",)
    assert graph.commands.pending("run-1") == ()

    rows = cast(
        list[tuple[object, ...]],
        graph_conn(graph)
        .execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        .fetchall(),
    )
    tables = {cast(str, row[0]) for row in rows}
    assert {"owner_capabilities", "session_commands", "command_receipts"} <= tables
    assert graph.get_node(node_id) is not None
    assert graph.get_all_nodes() == [graph.get_node(node_id)]
    graph.close()


def test_admission_is_idempotent_and_receipts_are_durable(graph: MikadoGraph) -> None:
    node_id = _owned_graph(graph)
    command = _command(node_id)

    first = graph.commands.admit(command, now=_NOW)
    duplicate = graph.commands.admit(command, now=_NOW)

    assert first.status == "queued"
    assert duplicate == first
    assert graph.commands.command(command.command_id) == command.__class__(
        command_id=command.command_id,
        node_id=node_id,
        run_id="run-1",
        invocation_id="invoke-1",
        owner_incarnation="owner-1",
        action="steer",
        text="redirect",
        expires_at=_LATER,
        status="queued",
        admitted_at=_NOW,
    )
    assert [receipt.status for receipt in graph.commands.history(command.command_id)] == ["queued"]
    with pytest.raises(ValueError, match="different command"):
        _ = graph.commands.admit(
            _command(node_id, text="changed"),
            now=_NOW,
        )


def test_admission_is_bounded_fifo_and_rejections_persist(graph: MikadoGraph) -> None:
    node_id = _owned_graph(graph)
    first = _command(node_id, command_id="first")
    second = _command(node_id, command_id="second")
    third = _command(node_id, command_id="third")

    assert graph.commands.admit(first, now=_NOW, max_pending=2).status == "queued"
    assert graph.commands.admit(second, now=_NOW, max_pending=2).status == "queued"
    rejected = graph.commands.admit(third, now=_NOW, max_pending=2)
    assert [command.command_id for command in graph.commands.pending("run-1", now=_NOW)] == [
        "first",
        "second",
    ]
    assert graph.commands.receipt("third") == rejected


def test_expiry_removes_queued_work_and_keeps_receipt(graph: MikadoGraph) -> None:
    node_id = _owned_graph(graph)
    command = _command(node_id, expires_at="2026-09-12T12:01:00+00:00")
    assert graph.commands.admit(command, now=_NOW).status == "queued"

    expired = graph.commands.expire(now="2026-09-12T12:01:00+00:00")

    assert [receipt.status for receipt in expired] == ["expired"]
    receipt = graph.commands.receipt(command.command_id)
    assert receipt is not None
    assert receipt.status == "expired"
    assert graph.commands.pending("run-1", now=_LATER) == ()


def test_attached_watch_admits_against_fresh_owner_capabilities(graph: MikadoGraph) -> None:
    node_id = _owned_graph(graph)
    admit = graph_command_admitter(graph)

    assert admit("run-1", SessionInput(action="steer", text="redirect")) is True
    pending = graph.commands.pending("run-1", now=_NOW)

    assert len(pending) == 1
    assert pending[0].node_id == node_id
    assert pending[0].action == "steer"


def test_admission_requires_current_owner_fence_and_exact_permission(graph: MikadoGraph) -> None:
    node_id = _owned_graph(graph)

    stale = graph.commands.admit(
        _command(node_id, command_id="stale", owner="old-owner"),
        now=_NOW,
    )
    wrong_permission = graph.commands.admit(
        _command(node_id, command_id="wrong", action="approve", permission_id="permission-2"),
        now=_NOW,
    )
    exact_permission = graph.commands.admit(
        _command(node_id, command_id="exact", action="approve", permission_id="permission-1"),
        now=_NOW,
    )

    assert stale.status == "rejected"
    assert "fence" in (stale.detail or "")
    assert wrong_permission.status == "rejected"
    assert wrong_permission.detail == "permission ID does not exactly match a current permission"
    assert exact_permission.status == "queued"


def test_transitions_require_all_fences_and_do_not_replay(graph: MikadoGraph) -> None:
    node_id = _owned_graph(graph)
    command = _command(node_id)
    assert graph.commands.admit(command, now=_NOW).status == "queued"

    submitted = graph.commands.submit(command, now=_NOW)
    delivered = graph.commands.deliver(command, now=_NOW)

    assert submitted.status == "submitted"
    assert delivered.status == "delivered"
    assert [receipt.status for receipt in graph.commands.history(command.command_id)] == [
        "queued",
        "submitted",
        "delivered",
    ]
    with pytest.raises(CommandFenceError, match="fence"):
        _ = graph.commands.transition(
            command.command_id,
            "unconfirmed",
            node_id=node_id,
            run_id="run-1",
            invocation_id="invoke-1",
            owner_incarnation="old-owner",
            now=_NOW,
        )
    with pytest.raises(ValueError, match="terminal"):
        _ = graph.commands.submit(command, now=_NOW)


def test_uncertain_delivery_is_terminal(graph: MikadoGraph) -> None:
    node_id = _owned_graph(graph)
    command = _command(node_id)
    assert graph.commands.admit(command, now=_NOW).status == "queued"

    uncertain = graph.commands.unconfirm(command, now=_NOW, detail="pipe closed")
    assert uncertain.status == "unconfirmed"
    with pytest.raises(ValueError, match="terminal"):
        _ = graph.commands.submit(command, now=_NOW)


def test_claim_is_atomic_across_graph_connections(tmp_path: Path) -> None:
    db_path = tmp_path / "claim.db"
    first = MikadoGraph(db_path)
    node_id = _owned_graph(first)
    command = _command(node_id)
    assert first.commands.admit(command, now=_NOW).status == "queued"
    second = MikadoGraph(db_path)

    try:
        claimed = first.commands.claim_pending("run-1", "owner-1", now=_NOW)
        competing = second.commands.claim_pending("run-1", "owner-1", now=_NOW)

        assert [item.command_id for item in claimed] == [command.command_id]
        assert competing == ()
        assert [receipt.status for receipt in first.commands.history(command.command_id)] == [
            "queued",
            "submitted",
        ]
    finally:
        second.close()
        first.close()


def test_claim_rejects_stale_owner_invocation(graph: MikadoGraph) -> None:
    node_id = _owned_graph(graph)
    command = _command(node_id)
    assert graph.commands.admit(command, now=_NOW).status == "queued"
    _ = graph.commands.publish_capabilities(
        "run-1",
        node_id,
        "invoke-2",
        "owner-1",
        ("steer",),
        published_at=_LATER,
    )

    assert graph.commands.claim_pending("run-1", "owner-1", now=_LATER) == ()
    stored = graph.commands.command(command.command_id)
    assert stored is not None and stored.status == "expired"
    assert [receipt.status for receipt in graph.commands.history(command.command_id)] == [
        "queued",
        "expired",
    ]


def test_existing_database_migrates_command_tables_without_node_backfill(tmp_path: Path) -> None:
    db_path = tmp_path / "legacy.db"
    graph = MikadoGraph(db_path)
    graph.close()
    conn = sqlite3.connect(db_path)
    _ = conn.execute("PRAGMA user_version = 4")
    conn.commit()
    conn.close()

    graph = MikadoGraph(db_path)
    rows = cast(
        list[tuple[object, ...]],
        graph_conn(graph)
        .execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        .fetchall(),
    )
    tables = {cast(str, row[0]) for row in rows}
    assert {"owner_capabilities", "session_commands", "command_receipts"} <= tables
    assert graph.get_all_nodes() == []
    graph.close()


def test_command_rejection_transition_and_id_generation(graph: MikadoGraph) -> None:
    node_id = _owned_graph(graph)
    command = _command(node_id)
    assert graph.commands.admit(command, now=_NOW).status == "queued"

    rejected = graph.commands.reject(command, now=_NOW, detail="worker refused")
    assert rejected.status == "rejected"
    assert len(new_command_id()) == 32
