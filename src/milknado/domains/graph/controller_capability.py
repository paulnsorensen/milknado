"""Out-of-band controller capability registration and one-use consumption."""

from __future__ import annotations

import hashlib
import hmac
import os
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from milknado.domains.common import CONTROLLER_MASTER_ENV

_LEDGER_NAME = "controller-capability.db"
_BINDING_PREFIX = b"milknado:goal-review:v1:"
_SCHEMA = """
CREATE TABLE IF NOT EXISTS controller_master (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    master_hash TEXT NOT NULL,
    registered_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS consumed_capability (
    capability_hash TEXT PRIMARY KEY,
    consumed_at TEXT NOT NULL
);
"""


def register_controller_master(project_root: Path) -> None:
    """Register the externally provisioned controller master before dispatch."""
    master_hash = _master_hash()
    if master_hash is None:
        return
    ledger = _ledger_path(project_root)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    if ledger.exists():
        ledger.chmod(0o600)
    with sqlite3.connect(ledger) as conn:
        _ = conn.executescript(_SCHEMA)
        _ = conn.execute(
            """
            INSERT OR IGNORE INTO controller_master(singleton, master_hash, registered_at)
            VALUES (1, ?, ?)
            """,
            (master_hash, _now()),
        )
        row = cast(
            tuple[str] | None,
            conn.execute(
                "SELECT master_hash FROM controller_master WHERE singleton = 1"
            ).fetchone(),
        )
        if row is None or not hmac.compare_digest(row[0], master_hash):
            raise RuntimeError("a different controller master is already registered")
        _ = conn.commit()
    ledger.chmod(0o600)


def consume_controller_capability(project_root: Path, review_id: int, decision: str) -> bool:
    """Atomically consume the controller capability for one review decision."""
    master = _master_bytes()
    ledger = _ledger_path(project_root)
    if master is None or not ledger.exists():
        return False
    capability_hash = _capability_hash(master, review_id, decision)
    try:
        with sqlite3.connect(ledger, timeout=5.0, isolation_level=None) as conn:
            _ = conn.execute("PRAGMA busy_timeout = 5000")
            _ = conn.execute("BEGIN IMMEDIATE")
            row = cast(
                tuple[str] | None,
                conn.execute(
                    "SELECT master_hash FROM controller_master WHERE singleton = 1"
                ).fetchone(),
            )
            if row is None or not hmac.compare_digest(row[0], _hash(master)):
                _ = conn.rollback()
                return False
            try:
                _ = conn.execute(
                    "INSERT INTO consumed_capability(capability_hash, consumed_at) VALUES (?, ?)",
                    (capability_hash, _now()),
                )
            except sqlite3.IntegrityError:
                _ = conn.rollback()
                return False
            _ = conn.commit()
            return True
    except sqlite3.Error:
        return False


def _ledger_path(project_root: Path) -> Path:
    return project_root.resolve() / ".milknado" / _LEDGER_NAME


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
    derived = hmac.new(master, binding, hashlib.sha256).digest()
    return _hash(derived)


def _now() -> str:
    return datetime.now(UTC).isoformat()
