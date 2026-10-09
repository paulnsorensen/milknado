from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from typing import cast

import msgspec

from milknado.domains.coordinator.model import ControlEvent

_MAX_CHAIN_DEPTH = 32
_ROLLOVER_MARKER = "[Earlier text omitted]\n"
_STREAMING_STATES = frozenset({"", "streaming", "running"})

EventRow = tuple[
    int,
    str,
    str,
    str,
    str,
    str,
    str,
    str,
    str,
    int | None,
    str,
    str | None,
    int | None,
    int | None,
]
EVENT_COLUMNS = (
    "seq, kind, text, entity_kind, entity_id, tool_name, status, "
    "turn_id, provider_session_id, duration_ms, created_at, "
    "stream_key, stream_ref, stream_depth"
)


@dataclass(frozen=True, slots=True)
class StoredText:
    seq: int
    session_id: str
    text: str
    status: str
    key: str | None
    ref: int | None
    depth: int | None


@dataclass(frozen=True, slots=True)
class StreamUpdate:
    key: str
    text: str
    status: str


@dataclass(frozen=True, slots=True)
class StreamFragment:
    text: str
    key: str
    ref: int | None
    depth: int


def stream_key(session_id: str, event: ControlEvent, native_id: str) -> str:
    identity = (
        session_id,
        event.turn_id,
        event.provider_session_id,
        event.kind,
        native_id,
    )
    return hashlib.sha256(msgspec.json.encode(identity)).hexdigest()


def _load(conn: sqlite3.Connection, seq: int) -> StoredText:
    row = cast(
        tuple[int, str, str, str, str | None, int | None, int | None] | None,
        conn.execute(
            "SELECT seq, session_id, text, status, stream_key, stream_ref, stream_depth "
            + "FROM coordinator_events WHERE seq = ?",
            (seq,),
        ).fetchone(),
    )
    if row is None:
        raise ValueError("stream history references a missing event")
    return StoredText(*row)


def _validated_root(row: StoredText, session_id: str, key: str | None) -> None:
    if row.session_id != session_id or row.key != key:
        raise ValueError("stream history crosses an owner or identity")
    if row.ref is None and row.depth != 0:
        raise ValueError("stream checkpoint has invalid depth")
    if row.ref is not None and (row.depth is None or not 1 <= row.depth <= _MAX_CHAIN_DEPTH):
        raise ValueError("stream suffix has invalid depth")


def decode_text(
    conn: sqlite3.Connection,
    session_id: str,
    row: StoredText,
    cache: dict[int, tuple[StoredText, str]],
) -> str:
    if row.key is None:
        if row.ref is not None or row.depth is not None or row.session_id != session_id:
            raise ValueError("ordinary event has stream metadata")
        return row.text
    chain: list[StoredText] = []
    current = row
    while current.seq not in cache:
        _validated_root(current, session_id, row.key)
        if current.ref is None:
            cache[current.seq] = (current, current.text)
            break
        if len(chain) >= _MAX_CHAIN_DEPTH or current.ref >= current.seq:
            raise ValueError("stream history exceeds checkpoint bound")
        previous = cache[current.ref][0] if current.ref in cache else _load(conn, current.ref)
        _validated_root(previous, session_id, row.key)
        if current.depth is None or previous.depth != current.depth - 1:
            raise ValueError("stream history has invalid depth")
        chain.append(current)
        current = previous
    cached, text = cache[current.seq]
    _validated_root(cached, session_id, row.key)
    for part in reversed(chain):
        text += part.text
        cache[part.seq] = (part, text)
    return text


def compact_stream(
    conn: sqlite3.Connection, session_id: str, update: StreamUpdate
) -> StreamFragment:
    row = cast(
        tuple[int] | None,
        conn.execute(
            "SELECT seq FROM coordinator_events WHERE session_id = ? AND stream_key = ? "
            + "ORDER BY seq DESC LIMIT 1",
            (session_id, update.key),
        ).fetchone(),
    )
    if row is not None and update.status in _STREAMING_STATES:
        previous = _load(conn, row[0])
        prior_text = decode_text(conn, session_id, previous, {})
        if (
            previous.status in _STREAMING_STATES
            and previous.depth is not None
            and previous.depth < _MAX_CHAIN_DEPTH
            and not update.text.startswith(_ROLLOVER_MARKER)
            and update.text.startswith(prior_text)
        ):
            return StreamFragment(
                update.text[len(prior_text) :], update.key, previous.seq, previous.depth + 1
            )
    return StreamFragment(update.text, update.key, None, 0)
