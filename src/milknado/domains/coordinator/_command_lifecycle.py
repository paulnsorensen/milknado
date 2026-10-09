from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Callable, Generator
from contextlib import contextmanager
from threading import Condition
from time import monotonic
from typing import Literal, cast

import msgspec

from milknado.domains.coordinator.commands import record_control_once
from milknado.domains.coordinator.control_models import (
    CoordinatorCommand,
    CoordinatorCommandReceipt,
    StartGoal,
)
from milknado.domains.coordinator.model import ControlEvent, CoordinatorSession
from milknado.domains.coordinator.receipt_results import receipt_payload


class CommandLifecycle:
    def __init__(self, connection: Callable[[], sqlite3.Connection]) -> None:
        self._connection: Callable[[], sqlite3.Connection] = connection
        self._done: Condition = Condition()
        self._closing: bool = False
        self._active: int = 0

    @contextmanager
    def command(self) -> Generator[None]:
        with self._done:
            if self._closing:
                raise RuntimeError("coordinator command is shutting down")
            self._active += 1
        try:
            yield
        finally:
            with self._done:
                self._active -= 1
                self._done.notify_all()

    def shutdown(self, stop_runtime: Callable[[], None], timeout: float = 10.0) -> None:
        with self._done:
            self._closing = True
        stop_runtime()
        deadline = monotonic() + timeout
        with self._done:
            while self._active and (remaining := deadline - monotonic()) > 0:
                _ = self._done.wait(remaining)
            if self._active:
                raise RuntimeError("coordinator command shutdown is unconfirmed")

    def complete(
        self,
        session_id: str,
        command: CoordinatorCommand,
        status: Literal["accepted", "unavailable", "unsupported", "rejected"],
        result: object,
    ) -> CoordinatorCommandReceipt:
        built = receipt_payload(result)
        command_id = command.command_id
        event_session_id = (
            cast(CoordinatorSession, result).id
            if isinstance(command, StartGoal) and status == "accepted"
            else session_id
        )
        conn = self._connection()
        with conn:
            _ = conn.execute(
                "UPDATE coordinator_web_receipts SET status = ?, result_json = ? "
                + "WHERE command_id = ?",
                (status, msgspec.json.encode(built).decode(), command_id),
            )
            if event_session_id:
                record_control_once(
                    conn,
                    event_session_id,
                    ControlEvent(
                        kind="command",
                        text=type(command).__name__,
                        entity_kind="coordinator_command",
                        entity_id=hashlib.sha256(command_id.encode()).hexdigest(),
                        status=status,
                    ),
                )
        return CoordinatorCommandReceipt(command_id, session_id, status, built)
