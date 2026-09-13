"""Graph-local controller registration and one-use review capabilities."""

from __future__ import annotations

import hashlib
import hmac
import os
import sqlite3
from datetime import UTC, datetime
from typing import cast

from milknado.domains.common.process import CONTROLLER_MASTER_ENV
from milknado.domains.graph._sqlite_rows import fetchone

CREATE_CONTROLLER_MASTER = (
    "CREATE TABLE IF NOT EXISTS controller_master ("
    "singleton INTEGER PRIMARY KEY CHECK (singleton = 1), "
    "master_hash TEXT NOT NULL, registered_at TEXT NOT NULL)"
)
CREATE_CONSUMED_CAPABILITY = (
    "CREATE TABLE IF NOT EXISTS consumed_controller_capabilities ("
    "capability_hash TEXT PRIMARY KEY, consumed_at TEXT NOT NULL)"
)
_BINDING_PREFIX = b"milknado:goal-review:v1:"


def register_controller_master(conn: sqlite3.Connection) -> None:
    """Register the externally provisioned controller master before dispatch."""
    master_hash = _master_hash()
    if master_hash is None:
        raise RuntimeError(f"{CONTROLLER_MASTER_ENV} is required before dispatch")
    _ = conn.execute(
        "INSERT OR IGNORE INTO controller_master (singleton, master_hash, registered_at) "
        + "VALUES (1, ?, ?)",
        (master_hash, _now()),
    )
    row = fetchone(conn, "SELECT master_hash FROM controller_master WHERE singleton = 1")
    if row is None or not hmac.compare_digest(cast(str, row[0]), master_hash):
        conn.rollback()
        raise RuntimeError("a different controller master is already registered")
    conn.commit()


def consume_controller_capability(conn: sqlite3.Connection, review_id: int, decision: str) -> bool:
    """Consume one review-bound capability inside the caller's graph transaction."""
    master = _master_bytes()
    row = fetchone(conn, "SELECT master_hash FROM controller_master WHERE singleton = 1")
    if master is None or row is None:
        return False
    if not hmac.compare_digest(cast(str, row[0]), _hash(master)):
        return False
    capability_hash = _capability_hash(master, review_id, decision)
    try:
        _ = conn.execute(
            "INSERT INTO consumed_controller_capabilities (capability_hash, consumed_at) "
            + "VALUES (?, ?)",
            (capability_hash, _now()),
        )
    except sqlite3.IntegrityError:
        return False
    return True


def _master_bytes() -> bytes | None:
    value = os.environ.get(CONTROLLER_MASTER_ENV, "")
    return value.encode() if value.strip() else None


def _master_hash() -> str | None:
    master = _master_bytes()
    return _hash(master) if master is not None else None


def _hash(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _capability_hash(master: bytes, review_id: int, decision: str) -> str:
    binding = _BINDING_PREFIX + f"{review_id}:{decision}".encode()
    return _hash(hmac.new(master, binding, hashlib.sha256).digest())


def _now() -> str:
    return datetime.now(UTC).isoformat()


__all__ = [
    "CREATE_CONSUMED_CAPABILITY",
    "CREATE_CONTROLLER_MASTER",
    "consume_controller_capability",
    "register_controller_master",
]
