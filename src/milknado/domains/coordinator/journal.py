# ruff: noqa: RUF100
from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import cast

import msgspec

from milknado.domains.common import redact_control_text as redact_control_text
from milknado.domains.coordinator._stream_history import (
    EVENT_COLUMNS,
    EventRow,
    StoredText,
    StreamFragment,
    StreamUpdate,
    compact_stream,
    decode_text,
    stream_key,
)
from milknado.domains.coordinator.model import ControlEvent, ControlRecord

_MAX_EVENT_BYTES = 64 * 1024
_MAX_DIAGNOSTIC_RETENTION = timedelta(days=30)
_DEFAULT_DIAGNOSTIC_RETENTION = timedelta(days=7)

_OPERATION_KINDS = frozenset(
    {
        "planning_decision",
        "graph_revision",
        "approval",
        "run_transition",
        "execution_group",
        "command",
    }
)


def operation_identity(event: ControlEvent) -> str | None:
    if event.kind not in _OPERATION_KINDS or not event.entity_kind or not event.entity_id:
        return None
    identity = (event.kind, event.entity_kind, event.entity_id, event.status)
    return hashlib.sha256(msgspec.json.encode(identity)).hexdigest()


def _utc(now: datetime | None) -> datetime:
    timestamp = now or datetime.now(UTC)
    if timestamp.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return timestamp.astimezone(UTC)


@dataclass(frozen=True, slots=True)
class _EventWrite:
    now: datetime | None = None
    native_id: str | None = None


@dataclass(frozen=True, slots=True)
class _StoredEvent:
    text: str
    timestamp: str
    expires_at: str | None
    fragment: StreamFragment | None


@dataclass(slots=True)
class _HistoryRead:
    cursor: int | None
    recovery_only: bool
    cache: dict[int, tuple[StoredText, str]]


def _record(
    conn: sqlite3.Connection,
    session_id: str,
    row: EventRow,
    cache: dict[int, tuple[StoredText, str]],
) -> ControlRecord:
    stored = StoredText(row[0], session_id, row[2], row[6], row[11], row[12], row[13])
    return ControlRecord(
        seq=row[0],
        kind=row[1],
        text=decode_text(conn, session_id, stored, cache),
        entity_kind=row[3],
        entity_id=row[4],
        tool_name=row[5],
        status=row[6],
        turn_id=row[7],
        provider_session_id=row[8],
        duration_ms=row[9],
        created_at=row[10],
    )


def _prepared_event(event: ControlEvent, now: datetime | None) -> tuple[str, str, str | None]:
    diagnostic_ttl = (
        timedelta(seconds=event.diagnostic_retention_seconds)
        if event.diagnostic_retention_seconds is not None
        else _DEFAULT_DIAGNOSTIC_RETENTION
    )
    if event.kind == "diagnostic" and not (
        timedelta(0) < diagnostic_ttl <= _MAX_DIAGNOSTIC_RETENTION
    ):
        raise ValueError("diagnostic retention must be between 0 and 30 days")
    if event.duration_ms is not None and event.duration_ms < 0:
        raise ValueError("duration_ms must not be negative")
    timestamp = _utc(now)
    text = "[tool payload elided]" if event.kind == "tool" else redact_control_text(event.text)
    if len(text.encode("utf-8")) > _MAX_EVENT_BYTES:
        raise ValueError("control event exceeds the 64 KiB limit")
    expires_at = (timestamp + diagnostic_ttl).isoformat() if event.kind == "diagnostic" else None
    return text, timestamp.isoformat(), expires_at


def _insert_event(
    conn: sqlite3.Connection, session_id: str, event: ControlEvent, stored: _StoredEvent
) -> int:
    fragment = stored.fragment
    cursor = conn.execute(
        "INSERT INTO coordinator_events "
        + "(session_id, kind, text, entity_kind, entity_id, tool_name, status, "
        + "turn_id, provider_session_id, duration_ms, created_at, expires_at, operation_hash, "
        + "stream_key, stream_ref, stream_depth) "
        + "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            session_id,
            event.kind,
            fragment.text if fragment else stored.text,
            redact_control_text(event.entity_kind),
            redact_control_text(event.entity_id),
            redact_control_text(event.tool_name),
            redact_control_text(event.status),
            redact_control_text(event.turn_id),
            redact_control_text(event.provider_session_id),
            event.duration_ms,
            stored.timestamp,
            stored.expires_at,
            operation_identity(event),
            fragment.key if fragment else None,
            fragment.ref if fragment else None,
            fragment.depth if fragment else None,
        ),
    )
    assert cursor.lastrowid is not None
    return cursor.lastrowid


def _write_event(
    conn: sqlite3.Connection, session_id: str, event: ControlEvent, options: _EventWrite
) -> int:
    if not session_id or not event.kind:
        raise ValueError("session_id and event kind must not be empty")
    text, timestamp, expires_at = _prepared_event(event, options.now)
    with conn:
        _ = conn.execute(
            "DELETE FROM coordinator_events WHERE expires_at IS NOT NULL AND expires_at <= ?",
            (timestamp,),
        )
        fragment = (
            compact_stream(
                conn,
                session_id,
                StreamUpdate(stream_key(session_id, event, options.native_id), text, event.status),
            )
            if options.native_id is not None
            else None
        )
        return _insert_event(
            conn, session_id, event, _StoredEvent(text, timestamp, expires_at, fragment)
        )


def append_control_event(
    conn: sqlite3.Connection,
    session_id: str,
    event: ControlEvent,
    *,
    now: datetime | None = None,
) -> int:
    return _write_event(conn, session_id, event, _EventWrite(now=now))


def append_stream_control_event(
    conn: sqlite3.Connection, session_id: str, event: ControlEvent, native_id: str
) -> int:
    if event.kind not in {"assistant", "error"} or not native_id:
        raise ValueError("stream storage requires an identified assistant or error event")
    return _write_event(conn, session_id, event, _EventWrite(native_id=native_id))


def control_history(
    conn: sqlite3.Connection,
    session_id: str,
    *,
    now: datetime | None = None,
    _read: _HistoryRead | None = None,
) -> tuple[ControlRecord, ...]:
    timestamp = _utc(now).isoformat()
    if _read is None:
        with conn:
            _ = conn.execute(
                "DELETE FROM coordinator_events WHERE expires_at IS NOT NULL AND expires_at <= ?",
                (timestamp,),
            )
    where = "WHERE session_id = ?"
    values: list[str | int] = [session_id]
    if _read is not None:
        if _read.cursor is not None:
            where += " AND seq > ?"
            values.append(_read.cursor)
        if _read.recovery_only:
            where += " AND kind = 'recovery'"
        where += " AND (expires_at IS NULL OR expires_at > ?)"
        values.append(timestamp)
    rows = cast(
        list[EventRow],
        conn.execute(
            f"SELECT {EVENT_COLUMNS} FROM coordinator_events {where} ORDER BY seq", values
        ).fetchall(),
    )
    cache = _read.cache if _read is not None else {}
    return tuple(_record(conn, session_id, row, cache) for row in rows)


def snapshot_control_history(
    conn: sqlite3.Connection, session_id: str, cursor: int, *, now: datetime | None = None
) -> tuple[tuple[ControlRecord, ...], tuple[ControlRecord, ...], int]:
    timestamp = _utc(now)
    cache: dict[int, tuple[StoredText, str]] = {}
    recent = control_history(
        conn, session_id, now=timestamp, _read=_HistoryRead(cursor, False, cache)
    )
    recovery = control_history(
        conn, session_id, now=timestamp, _read=_HistoryRead(None, True, cache)
    )
    if recent:
        latest = recent[-1].seq
    else:
        row = cast(
            tuple[int | None],
            conn.execute(
                "SELECT MAX(seq) FROM coordinator_events WHERE session_id = ? "
                + "AND (expires_at IS NULL OR expires_at > ?)",
                (session_id, timestamp.isoformat()),
            ).fetchone(),
        )
        latest = row[0] if row[0] is not None else cursor
    return recent, recovery, latest
