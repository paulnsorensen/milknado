# ruff: noqa: RUF100
from __future__ import annotations

import re
import sqlite3
from datetime import UTC, datetime, timedelta
from typing import cast

from milknado.domains.coordinator.model import ControlEvent, ControlRecord

_MAX_EVENT_BYTES = 64 * 1024
_MAX_DIAGNOSTIC_RETENTION = timedelta(days=30)
_DEFAULT_DIAGNOSTIC_RETENTION = timedelta(days=7)
_QUOTED_START = re.compile(
    r"(?i)((?:[\"']?(?:api[_-]?key|password|token|client[_-]?secret|secret|"
    + r"authorization)[\"']?[ \t]*[:=][ \t]*|bearer[ \t]+))([\"'])"
)
_AUTH_HEADER = re.compile(r"(?im)(\bauthorization[ \t]*[:=][ \t]*)[^\r\n]*")
_SECRET = re.compile(
    r"(?i)(\b(?:bearer[ \t]+|api[_-]?key[ \t]*[=:][ \t]*|"
    + r"password[ \t]*[=:][ \t]*|token[ \t]*[=:][ \t]*|"
    + r"client[_-]?secret[ \t]*[=:][ \t]*|secret[ \t]*[=:][ \t]*))"
    + r"(?![\"'])[^\s,;]+"
    + r"|\b(?:sk-[A-Za-z0-9_-]{8,}|ghp_[A-Za-z0-9_]{8,})\b"
)


def _redact_quoted(value: str) -> str:
    parts: list[str] = []
    position = 0
    while match := _QUOTED_START.search(value, position):
        parts.append(value[position : match.end()])
        quote = match.group(2)
        cursor = match.end()
        while cursor < len(value):
            if value[cursor] == "\\":
                cursor += 2
            elif value[cursor] == quote:
                break
            else:
                cursor += 1
        parts.append("[REDACTED]")
        if cursor >= len(value):
            position = cursor
            break
        parts.append(quote)
        position = cursor + 1
    parts.append(value[position:])
    return "".join(parts)


def _redact(value: str) -> str:
    quoted = _redact_quoted(value)
    headers = _AUTH_HEADER.sub(lambda match: match.group(1) + "[REDACTED]", quoted)
    return _SECRET.sub(lambda match: (match.group(1) or "") + "[REDACTED]", headers)


def _utc(now: datetime | None) -> datetime:
    timestamp = now or datetime.now(UTC)
    if timestamp.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return timestamp.astimezone(UTC)


def _record(row: tuple[int, str, str, str, str, str, str, int | None, str]) -> ControlRecord:
    return ControlRecord(
        seq=row[0],
        kind=str(row[1]),
        text=str(row[2]),
        entity_kind=str(row[3]),
        entity_id=str(row[4]),
        tool_name=str(row[5]),
        status=str(row[6]),
        duration_ms=row[7],
        created_at=str(row[8]),
    )


def append_control_event(  # noqa
    conn: sqlite3.Connection,
    session_id: str,
    event: ControlEvent,
    *,
    now: datetime | None = None,
) -> int:
    if not session_id or not event.kind:
        raise ValueError("session_id and event kind must not be empty")
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
    text = "[tool payload elided]" if event.kind == "tool" else _redact(event.text)
    if len(text.encode("utf-8")) > _MAX_EVENT_BYTES:
        raise ValueError("control event exceeds the 64 KiB limit")
    expires_at = (timestamp + diagnostic_ttl).isoformat() if event.kind == "diagnostic" else None
    with conn:
        _ = conn.execute(
            "DELETE FROM coordinator_events WHERE expires_at IS NOT NULL AND expires_at <= ?",
            (timestamp.isoformat(),),
        )
        cursor = conn.execute(
            "INSERT INTO coordinator_events "
            + "(session_id, kind, text, entity_kind, entity_id, tool_name, status, "
            + "duration_ms, created_at, expires_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                session_id,
                event.kind,
                text,
                _redact(event.entity_kind),
                _redact(event.entity_id),
                _redact(event.tool_name),
                _redact(event.status),
                event.duration_ms,
                timestamp.isoformat(),
                expires_at,
            ),
        )
    assert cursor.lastrowid is not None
    return cursor.lastrowid


def control_history(  # noqa
    conn: sqlite3.Connection, session_id: str, *, now: datetime | None = None
) -> tuple[ControlRecord, ...]:
    timestamp = _utc(now).isoformat()
    with conn:
        _ = conn.execute(
            "DELETE FROM coordinator_events WHERE expires_at IS NOT NULL AND expires_at <= ?",
            (timestamp,),
        )
    rows = cast(
        list[tuple[int, str, str, str, str, str, str, int | None, str]],
        conn.execute(
            "SELECT seq, kind, text, entity_kind, entity_id, tool_name, status, duration_ms, "
            + "created_at FROM coordinator_events WHERE session_id = ? ORDER BY seq",
            (session_id,),
        ).fetchall(),
    )
    return tuple(_record(row) for row in rows)


def snapshot_control_history(
    conn: sqlite3.Connection, session_id: str, *, now: datetime | None = None
) -> tuple[ControlRecord, ...]:
    timestamp = _utc(now).isoformat()
    rows = cast(
        list[tuple[int, str, str, str, str, str, str, int | None, str]],
        conn.execute(
            "SELECT seq, kind, text, entity_kind, entity_id, tool_name, status, duration_ms, "
            + "created_at FROM coordinator_events WHERE session_id = ? "
            + "AND (expires_at IS NULL OR expires_at > ?) ORDER BY seq",
            (session_id, timestamp),
        ).fetchall(),
    )
    return tuple(_record(row) for row in rows)
