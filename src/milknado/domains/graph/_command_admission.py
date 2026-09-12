"""Shared admission service for live and attached session commands."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import msgspec

from milknado.domains.common import SessionInput
from milknado.domains.graph.commands import GraphCommand, new_command_id
from milknado.domains.graph.graph import MikadoGraph


@dataclass(frozen=True, slots=True)
class _AdmissionLookup:
    found: bool
    command: SessionInput | None = None


def admit_session_command(
    graph: MikadoGraph,
    run_id: str,
    command: SessionInput,
    *,
    owner_incarnation: str | None = None,
) -> SessionInput | None:
    """Persist one command against the current owner capability fence."""
    command_id = (
        command.command_id
        or (new_command_id() if command.action in {"approve", "deny"} else command.request_id)
        or new_command_id()
    )
    known = _known_admission(graph, run_id, command, owner_incarnation)
    if known.found:
        return known.command

    record = graph.runs.get(run_id)
    capabilities = graph.commands.capabilities(run_id)
    _ = graph.commands.expire()
    if record is None or capabilities is None or command.action not in capabilities.actions:
        return None
    if owner_incarnation is not None and owner_incarnation != capabilities.owner_incarnation:
        return None
    if command.owner_incarnation and command.owner_incarnation != capabilities.owner_incarnation:
        return None
    if command.invocation_id and command.invocation_id != capabilities.invocation_id:
        return None
    permission_id = command.request_id if command.action in {"approve", "deny"} else None
    if permission_id is not None and permission_id not in capabilities.permission_ids:
        return None
    value = GraphCommand(
        command_id=command_id,
        node_id=record["node_id"],
        run_id=run_id,
        invocation_id=capabilities.invocation_id,
        owner_incarnation=capabilities.owner_incarnation,
        action=command.action,
        text=command.text,
        permission_id=permission_id,
        expires_at=(datetime.now(UTC) + timedelta(hours=1)).isoformat(),
    )
    try:
        receipt = graph.commands.admit(value)
    except ValueError:
        known = _known_admission(graph, run_id, command, owner_incarnation)
        if known.found:
            return known.command
        raise
    if receipt.status != "queued":
        return None
    return msgspec.structs.replace(command, command_id=command_id)


def _known_admission(
    graph: MikadoGraph,
    run_id: str,
    command: SessionInput,
    owner_incarnation: str | None,
) -> _AdmissionLookup:
    command_id = command.command_id or (
        command.request_id if command.action not in {"approve", "deny"} else ""
    )
    if not command_id:
        return _AdmissionLookup(False)
    stored = graph.commands.command(command_id)
    if stored is None:
        return _AdmissionLookup(False)
    permission_id = command.request_id if command.action in {"approve", "deny"} else None
    if (
        stored.run_id != run_id
        or stored.action != command.action
        or stored.text != command.text
        or stored.permission_id != permission_id
    ):
        raise ValueError("command_id already names a different command")
    if any(
        fence and fence != stored.owner_incarnation
        for fence in (owner_incarnation, command.owner_incarnation)
    ) or (command.invocation_id and command.invocation_id != stored.invocation_id):
        return _AdmissionLookup(True)
    if stored.status not in {"queued", "submitted", "delivered"}:
        return _AdmissionLookup(True)
    if graph.commands.receipt(command_id) is None:
        raise RuntimeError("stored command has no receipt")
    return _AdmissionLookup(True, msgspec.structs.replace(command, command_id=command_id))
