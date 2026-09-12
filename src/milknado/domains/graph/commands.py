"""Immutable records for the graph-owned session command inbox."""

from __future__ import annotations

import uuid
from typing import Literal

import msgspec

from milknado.domains.common.session import SessionAction

CommandStatus = Literal[
    "queued",
    "submitted",
    "delivered",
    "rejected",
    "expired",
    "unconfirmed",
]


class GraphCommand(msgspec.Struct, frozen=True, kw_only=True):
    """A command admitted for one exact owner invocation."""

    command_id: str
    node_id: int
    run_id: str
    invocation_id: str
    owner_incarnation: str
    action: SessionAction
    expires_at: str
    text: str = ""
    permission_id: str | None = None
    status: CommandStatus = "queued"
    admitted_at: str = ""  # noqa: V107 - populated by durable command storage


class CommandReceipt(msgspec.Struct, frozen=True, kw_only=True):
    """Durable observation of one command state."""

    command_id: str
    status: CommandStatus
    node_id: int
    run_id: str
    invocation_id: str
    owner_incarnation: str
    action: SessionAction
    text: str
    permission_id: str | None
    expires_at: str
    admitted_at: str  # noqa: V107 - populated by durable command storage
    recorded_at: str  # noqa: V107 - populated by durable receipt storage
    detail: str | None = None


class OwnerCapabilities(msgspec.Struct, frozen=True, kw_only=True):
    """Fresh capabilities published by the current live session owner."""

    run_id: str
    node_id: int
    invocation_id: str
    owner_incarnation: str
    actions: tuple[SessionAction, ...]
    permission_ids: tuple[str, ...]
    published_at: str


class CommandFenceError(ValueError):
    """Raised when a command no longer matches the current owner fence."""


def new_command_id() -> str:
    """Return a collision-resistant stable ID for a caller-created command."""
    return uuid.uuid4().hex


__all__ = [
    "CommandFenceError",
    "CommandReceipt",
    "CommandStatus",
    "GraphCommand",
    "OwnerCapabilities",
    "new_command_id",
]
