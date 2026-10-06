"""Submit a coordinator runtime action without holding the graph lock."""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Callable
from contextlib import closing
from typing import Literal

import msgspec

from milknado.domains.coordinator.commands import CoordinatorAction, submit_coordinator_action
from milknado.domains.coordinator.control_models import (
    CoordinatorCommandReceipt,
    RuntimeAction,
)
from milknado.domains.coordinator.control_services import CoordinatorServices
from milknado.domains.coordinator.persistence import create_coordinator_tables, get_coordinator
from milknado.domains.coordinator.receipt_results import reserve_command_receipt
from milknado.domains.graph import MikadoGraph


def send_runtime_action(  # noqa: PLR0913 - command completion belongs to the controller
    graph: MikadoGraph,
    services: CoordinatorServices,
    session_id: str,
    command: RuntimeAction,
    complete: Callable[..., CoordinatorCommandReceipt],
) -> CoordinatorCommandReceipt:
    with graph.synchronization_lock:
        conn = graph.group_connection
        create_coordinator_tables(conn)
        session = get_coordinator(conn, session_id)
        if session is None:
            raise KeyError(session_id)
        fingerprint = hashlib.sha256(msgspec.json.encode(command)).hexdigest()
        existing = reserve_command_receipt(conn, session_id, command.command_id, fingerprint)
        if existing is not None:
            return existing
        if services.runtime_session is None:
            return complete(
                session_id, command, "unavailable", "Provider runtime is not connected."
            )
        runtime = services.runtime_session(command.provider_session_id)
        if runtime is None:
            return complete(session_id, command, "unavailable", "Provider session is not active.")
    try:
        with closing(sqlite3.connect(graph.db_path)) as conn:
            result = submit_coordinator_action(
                conn, session, runtime, CoordinatorAction(command.command_id, command.input)
            )
        status: Literal["accepted", "rejected"] = "accepted"
    except (ValueError, PermissionError) as error:
        status, result = "rejected", str(error)
    with graph.synchronization_lock:
        return complete(session_id, command, status, result)
