"""Durable structured-session context and event persistence."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from typing import cast

import msgspec

import milknado.domains.graph._run_persistence as _run_persistence
from milknado.domains.common import (
    SessionContext,
    SessionEvent,
    SessionView,
    normalize_session_event,
)
from milknado.domains.graph._sqlite_rows import fetchall, fetchone

_SESSION_ROLE = "session"
_MAX_SESSION_VIEW = 500
_MAX_EVENT_BYTES = 64 * 1024

_LATEST_EVENT_ROWS_SQL = """
WITH valid AS (
    SELECT seq, body,
           ROW_NUMBER() OVER (
               PARTITION BY json_extract(body, '$.kind'),
               COALESCE(json_extract(body, '$.event_id'), ''),
               CASE WHEN COALESCE(json_extract(body, '$.event_id'), '') = ''
                    THEN seq END
               ORDER BY seq DESC
           ) AS row_number
    FROM run_messages
    WHERE run_id = ? AND role = ? AND json_valid(body)
),
candidates AS (
    SELECT seq, body FROM valid WHERE row_number = 1
    UNION ALL
    SELECT seq, body FROM run_messages
    WHERE run_id = ? AND role = ? AND NOT json_valid(body)
)
SELECT seq, body FROM candidates ORDER BY seq DESC LIMIT ?
"""


def _validate_limit(limit: int) -> None:
    if not 0 <= limit <= _MAX_SESSION_VIEW:
        raise ValueError(f"limit must be between 0 and {_MAX_SESSION_VIEW}")


def _decode_event(body: str, seq: int) -> SessionEvent:
    try:
        return msgspec.json.decode(body, type=SessionEvent)
    except (msgspec.DecodeError, msgspec.ValidationError, TypeError, ValueError) as exc:
        return SessionEvent(
            kind="error",
            text=f"Stored session event {seq} is invalid: {exc}",
            event_id=f"persisted-{seq}",
            state="failed",
        )


def _latest_event(
    conn: sqlite3.Connection, run_id: str, event: SessionEvent
) -> SessionEvent | None:
    if not event.event_id:
        return None
    row = fetchone(
        conn,
        "SELECT seq, body FROM run_messages "
        + "WHERE run_id = ? AND role = ? AND json_valid(body) "
        + "AND json_extract(body, '$.kind') = ? AND json_extract(body, '$.event_id') = ? "
        + "ORDER BY seq DESC LIMIT 1",
        (run_id, _SESSION_ROLE, event.kind, event.event_id),
    )
    return _decode_event(cast(str, row[1]), cast(int, row[0])) if row is not None else None


def start_session(conn: sqlite3.Connection, run_id: str, context: SessionContext) -> None:
    """Persist one immutable context before a worker session starts."""
    if not run_id:
        raise ValueError("run_id must not be empty")
    with conn:
        _ = conn.execute(
            "INSERT INTO run_sessions (run_id, family, cwd, base_oid) VALUES (?, ?, ?, ?)",
            (run_id, context.family, context.cwd, context.base_oid),
        )


def append_session_event(conn: sqlite3.Connection, run_id: str, event: SessionEvent) -> int:
    """Append a normalized event snapshot using the run-message sequence."""
    if not run_id:
        raise ValueError("run_id must not be empty")
    _ = conn.execute("BEGIN IMMEDIATE")
    try:
        previous = (
            _latest_event(conn, run_id, event) if event.delta or event.kind == "user" else None
        )
        normalized = normalize_session_event(event, previous)
        body = msgspec.json.encode(normalized)
        if len(body) > _MAX_EVENT_BYTES:
            raise ValueError("session event exceeds the 64 KiB durable message limit")
        return _run_persistence.deposit_run_message(
            conn,
            run_id,
            _SESSION_ROLE,
            body.decode("utf-8"),
            datetime.now(UTC).isoformat(),
        )
    except Exception:
        conn.rollback()
        raise


def _context(conn: sqlite3.Connection, run_id: str) -> SessionContext | None:
    table = fetchone(
        conn, "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'run_sessions'"
    )
    if table is None:
        return None
    row = fetchone(
        conn, "SELECT family, cwd, base_oid FROM run_sessions WHERE run_id = ?", (run_id,)
    )
    if row is None:
        return None
    return msgspec.convert(
        {"family": row[0], "cwd": row[1], "base_oid": row[2]},
        type=SessionContext,
        strict=True,
    )


def _events(conn: sqlite3.Connection, run_id: str, limit: int) -> tuple[SessionEvent, ...]:
    if limit == 0:
        return ()
    rows = fetchall(
        conn,
        _LATEST_EVENT_ROWS_SQL,
        (run_id, _SESSION_ROLE, run_id, _SESSION_ROLE, limit),
    )
    return tuple(_decode_event(cast(str, row[1]), cast(int, row[0])) for row in reversed(rows))


def view_session(
    conn: sqlite3.Connection,
    run_id: str,
    limit: int = _MAX_SESSION_VIEW,
    *,
    active: bool | None = None,
) -> SessionView:
    """Read a bounded, coalesced session projection from the graph connection."""
    _validate_limit(limit)
    context = _context(conn, run_id)
    if active is None:
        row = fetchone(conn, "SELECT status FROM runs WHERE run_id = ?", (run_id,))
        active = row is not None and cast(str, row[0]) == "running"
    events = _events(conn, run_id, limit)
    return SessionView(
        context=context,
        events=events,
        active=bool(context is not None and active),
        permissions=tuple(
            event for event in events if event.kind == "permission" and event.state == "requested"
        ),
    )


__all__ = ["append_session_event", "start_session", "view_session"]
