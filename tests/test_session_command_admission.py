from __future__ import annotations

import pytest

from milknado.domains.common import SessionContext, SessionInput
from milknado.domains.graph import MikadoGraph
from milknado.domains.graph._command_admission import admit_session_command
from tests.graph_command_fixtures import command as _command

_NOW = "2026-09-12T12:00:00+00:00"


def _ready_graph(graph: MikadoGraph) -> None:
    node = graph.add_node("steerable")
    assert graph.claim_node(node.id, "run-1", now=_NOW)
    graph.runs.start("run-1", node.id, "run.log", _NOW, 60)
    _ = graph.commands.publish_capabilities(
        "run-1", node.id, "invoke-1", "owner-1", ("approve",), ("permission-1",), published_at=_NOW
    )


def test_duplicate_admission_preserves_expiry_and_receipt(graph: MikadoGraph) -> None:
    _ready_graph(graph)
    command = SessionInput(
        action="approve", request_id="permission-1", command_id="stable-command"
    )
    admitted = admit_session_command(graph, "run-1", command, owner_incarnation="owner-1")
    assert admitted == command
    original = graph.commands.command("stable-command")
    history = graph.commands.history("stable-command")

    duplicate = admit_session_command(graph, "run-1", command, owner_incarnation="owner-1")

    assert duplicate == command
    assert graph.commands.command("stable-command") == original
    assert graph.commands.history("stable-command") == history


def test_duplicate_submitted_and_delivered_is_not_replayed(graph: MikadoGraph) -> None:
    _ready_graph(graph)
    command = SessionInput(
        action="approve", request_id="permission-1", command_id="stable-command"
    )
    assert admit_session_command(graph, "run-1", command) == command
    stored = graph.commands.command("stable-command")
    assert stored is not None
    assert graph.commands.submit(stored).status == "submitted"
    assert admit_session_command(graph, "run-1", command) == command
    stored = graph.commands.command("stable-command")
    assert stored is not None
    assert graph.commands.deliver(stored).status == "delivered"
    assert admit_session_command(graph, "run-1", command) == command
    assert [receipt.status for receipt in graph.commands.history("stable-command")] == [
        "queued",
        "submitted",
        "delivered",
    ]


def test_duplicate_does_not_require_current_permission_capability(graph: MikadoGraph) -> None:
    _ready_graph(graph)
    command = SessionInput(
        action="approve", request_id="permission-1", command_id="stable-command"
    )
    assert admit_session_command(graph, "run-1", command) == command
    _ = graph.commands.publish_capabilities(
        "run-1", 1, "invoke-2", "owner-1", ("approve",), (), published_at=_NOW
    )

    assert admit_session_command(graph, "run-1", command, owner_incarnation="owner-1") == command


def test_reused_command_id_with_changed_payload_is_rejected(graph: MikadoGraph) -> None:
    _ready_graph(graph)
    command = SessionInput(
        action="approve", request_id="permission-1", command_id="stable-command"
    )
    assert admit_session_command(graph, "run-1", command) == command

    with pytest.raises(ValueError, match="different command"):
        _ = admit_session_command(
            graph,
            "run-1",
            SessionInput(
                action="approve",
                request_id="permission-1",
                text="changed",
                command_id="stable-command",
            ),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [("owner_incarnation", "owner-2"), ("invocation_id", "invoke-2")],
)
def test_known_command_rejects_supplied_fence_without_adapter_owner(
    graph: MikadoGraph, field: str, value: str
) -> None:
    _ready_graph(graph)
    original = SessionInput(
        action="approve", request_id="permission-1", command_id="stable-command"
    )
    assert admit_session_command(graph, "run-1", original) == original
    stored = graph.commands.command("stable-command")
    history = graph.commands.history("stable-command")
    retry = SessionInput(
        action="approve",
        request_id="permission-1",
        command_id="stable-command",
        **{field: value},
    )

    assert admit_session_command(graph, "run-1", retry) is None
    assert graph.commands.command("stable-command") == stored
    assert graph.commands.history("stable-command") == history


@pytest.mark.parametrize(
    ("owner_incarnation", "invocation_id"),
    [("owner-1", "invoke-2"), ("owner-2", "invoke-1")],
)
def test_displayed_fence_rejects_stale_session_input(
    graph: MikadoGraph, owner_incarnation: str, invocation_id: str
) -> None:
    _ready_graph(graph)
    _ = graph.commands.publish_capabilities(
        "run-1", 1, "invoke-2", "owner-2", ("approve",), ("permission-1",), published_at=_NOW
    )
    command = SessionInput(
        action="approve",
        request_id="permission-1",
        command_id="stale-command",
        owner_incarnation=owner_incarnation,
        invocation_id=invocation_id,
    )

    assert admit_session_command(graph, "run-1", command) is None
    assert graph.commands.command("stale-command") is None


def test_session_view_projects_current_fence(graph: MikadoGraph) -> None:
    _ready_graph(graph)
    graph.sessions.start("run-1", SessionContext(family="omp", cwd="/repo"))

    view = graph.sessions.view("run-1")

    assert (view.owner_incarnation, view.invocation_id) == ("owner-1", "invoke-1")


def test_duplicate_rejects_replacement_fences_without_mutation(graph: MikadoGraph) -> None:
    _ready_graph(graph)
    command = SessionInput(
        action="approve", request_id="permission-1", command_id="stable-command"
    )
    assert admit_session_command(graph, "run-1", command, owner_incarnation="owner-1") == command
    original = graph.commands.command("stable-command")
    history = graph.commands.history("stable-command")
    _ = graph.commands.publish_capabilities(
        "run-1", 1, "invoke-2", "owner-2", ("approve",), ("permission-1",), published_at=_NOW
    )

    replacement = SessionInput(
        action="approve",
        request_id="permission-1",
        command_id="stable-command",
        owner_incarnation="owner-2",
        invocation_id="invoke-2",
    )

    assert admit_session_command(graph, "run-1", replacement, owner_incarnation="owner-2") is None
    assert graph.commands.command("stable-command") == original
    assert graph.commands.history("stable-command") == history


@pytest.mark.parametrize("status", ["rejected", "expired", "unconfirmed"])
def test_terminal_duplicate_returns_durable_non_success(graph: MikadoGraph, status: str) -> None:
    _ready_graph(graph)
    command = SessionInput(
        action="approve",
        text="redirect",
        request_id="permission-1",
        command_id=f"{status}-command",
    )
    base = {
        "action": "approve",
        "permission_id": "permission-1",
        "command_id": command.command_id,
    }
    if status == "rejected":
        base["owner"] = "old-owner"
    elif status == "expired":
        base["expires_at"] = "2026-09-12T11:00:00+00:00"
    _ = graph.commands.admit(_command(1, **base), now=_NOW)
    if status == "unconfirmed":
        stored = graph.commands.command(command.command_id)
        assert stored is not None
        _ = graph.commands.unconfirm(stored, now=_NOW, detail="delivery uncertain")
    original = graph.commands.command(command.command_id)
    history = graph.commands.history(command.command_id)

    assert admit_session_command(graph, "run-1", command, owner_incarnation="owner-1") is None
    assert graph.commands.command(command.command_id) == original
    assert graph.commands.history(command.command_id) == history
