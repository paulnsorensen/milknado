"""Graph-local controller registration and one-use review capabilities."""

from __future__ import annotations

import contextlib
import hashlib
import hmac
import os
import re
import secrets
import sqlite3
import stat
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from milknado.domains.common.process import (
    CONTROLLER_MASTER_ENV,
    WORKER_CONTEXT_ENV,
)
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
_HASH_RE = re.compile(r"[0-9a-f]{64}\Z")


class ControllerAuthorizationError(RuntimeError):
    """Raised when controller authorization or credential storage fails."""


def register_controller_master(conn: sqlite3.Connection) -> bytes:
    """Resolve and register the controller credential for a graph."""
    if os.environ.get(WORKER_CONTEXT_ENV) == "1":
        raise ControllerAuthorizationError(
            "controller authorization is unavailable in a worker context"
        )
    _ = conn.execute("BEGIN IMMEDIATE")
    try:
        row = fetchone(conn, "SELECT master_hash FROM controller_master WHERE singleton = 1")
        registered_hash = cast(str, row[0]) if row is not None else None
        explicit = _explicit_master()
        if registered_hash is not None:
            master_hash = _validated_hash(registered_hash)
            master = explicit or _load_credential(master_hash)
            if explicit is not None and _hash(explicit) != master_hash:
                raise ControllerAuthorizationError(
                    "a different controller master is already registered; use the original "
                    + f"{CONTROLLER_MASTER_ENV} or restore its credential backup"
                )
            if master is None:
                raise ControllerAuthorizationError(
                    "the registered controller credential is unavailable; set "
                    + f"{CONTROLLER_MASTER_ENV} to the original secret or restore "
                    + f"the credential backup under {_store_dir()}"
                )
            _publish_if_missing(master_hash, master)
        else:
            master = explicit or secrets.token_bytes(32)
            master_hash = _hash(master)
            _publish_if_missing(master_hash, master)
            _ = conn.execute(
                "INSERT INTO controller_master (singleton, master_hash, registered_at) "
                + "VALUES (1, ?, ?)",
                (master_hash, _now()),
            )
        conn.commit()
        return master
    except Exception:
        conn.rollback()
        raise


def consume_controller_capability(
    conn: sqlite3.Connection,
    review_id: int,
    decision: str,
    master: bytes | None = None,
) -> bool:
    """Consume one review-bound capability inside the caller's graph transaction."""
    if os.environ.get(WORKER_CONTEXT_ENV) == "1":
        return False
    explicit = _explicit_master()
    credential = explicit if explicit is not None else master
    row = fetchone(conn, "SELECT master_hash FROM controller_master WHERE singleton = 1")
    if credential is None or row is None:
        return False
    if not hmac.compare_digest(cast(str, row[0]), _hash(credential)):
        return False
    capability_hash = _capability_hash(credential, review_id, decision)
    try:
        _ = conn.execute(
            "INSERT INTO consumed_controller_capabilities (capability_hash, consumed_at) "
            + "VALUES (?, ?)",
            (capability_hash, _now()),
        )
    except sqlite3.IntegrityError:
        return False
    return True


def _explicit_master() -> bytes | None:
    """Read and validate an explicitly configured controller credential."""
    if CONTROLLER_MASTER_ENV not in os.environ:
        return None
    value = os.environ[CONTROLLER_MASTER_ENV]
    if not value.strip():
        raise ControllerAuthorizationError(
            f"{CONTROLLER_MASTER_ENV} is set but empty; unset it or provide the original secret"
        )
    master = value.encode()
    if len(master) > 4096:
        raise ControllerAuthorizationError(f"{CONTROLLER_MASTER_ENV} exceeds the 4096-byte limit")
    return master


def _load_credential(master_hash: str) -> bytes | None:
    """Load the owner-only credential matching a registered hash, if present."""
    path = _credential_path(master_hash)
    try:
        if path.is_symlink():
            raise ControllerAuthorizationError(
                f"controller credential record is a symlink: {path}"
            )
        flags = os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(path, flags)
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
                raise ControllerAuthorizationError(
                    f"controller credential record is not owner-owned: {path}"
                )
            if stat.S_IMODE(info.st_mode) & 0o077 or info.st_size > 4096:
                raise ControllerAuthorizationError(
                    f"controller credential record has unsafe permissions: {path}"
                )
            value = os.read(fd, 4097)
        finally:
            os.close(fd)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise ControllerAuthorizationError(
            f"cannot read controller credential store at {path}: {exc}"
        ) from exc
    if not value or _hash(value) != master_hash:
        raise ControllerAuthorizationError(f"controller credential record is corrupt: {path}")
    return value


def _publish_if_missing(master_hash: str, master: bytes) -> None:
    """Atomically persist a controller credential when no record exists."""
    path = _credential_path(master_hash)
    existing = _load_credential(master_hash)
    if existing is not None:
        return
    store = path.parent
    temporary = ""
    try:
        fd, temporary = tempfile.mkstemp(prefix=".credential-", dir=store)
        with os.fdopen(fd, "wb") as stream:
            os.fchmod(stream.fileno(), 0o600)
            _ = stream.write(master)
            stream.flush()
            os.fsync(stream.fileno())
        _ = Path(temporary).replace(path)
        directory_fd = os.open(store, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError as exc:
        with contextlib.suppress(OSError):
            Path(temporary).unlink()
        raise ControllerAuthorizationError(
            f"cannot publish controller credential store at {store}: {exc}"
        ) from exc


def _credential_path(master_hash: str) -> Path:
    """Return the managed record path for a controller hash."""
    return _store_dir() / master_hash


def _store_dir() -> Path:
    """Return the owner-only managed controller credential directory."""
    if os.name == "nt":
        raise ControllerAuthorizationError(
            "managed controller credentials require a POSIX platform; Windows is unsupported"
        )
    state_home = os.environ.get("XDG_STATE_HOME", "").strip()
    root = Path(state_home) if state_home else Path.home() / ".local" / "state"
    if state_home and not root.is_absolute():
        raise ControllerAuthorizationError("XDG_STATE_HOME must be an absolute path")
    namespace = root / "milknado"
    store = namespace / "controllers"
    try:
        for directory in (root, namespace, store):
            if directory.is_symlink():
                raise ControllerAuthorizationError(
                    f"controller credential store contains a symlink: {directory}"
                )
        root.mkdir(parents=True, exist_ok=True)
        namespace.mkdir(mode=0o700, exist_ok=True)
        store.mkdir(mode=0o700, exist_ok=True)
        for directory in (namespace, store):
            info = directory.stat()
            if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
                raise ControllerAuthorizationError(
                    f"controller credential store is not owner-only: {directory}"
                )
    except OSError as exc:
        raise ControllerAuthorizationError(
            f"cannot access controller credential store at {store}: {exc}"
        ) from exc
    return store


def _validated_hash(value: str) -> str:
    """Reject malformed controller hashes read from graph storage."""
    if not _HASH_RE.fullmatch(value):
        raise ControllerAuthorizationError("registered controller hash is malformed")
    return value


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
    "ControllerAuthorizationError",
    "consume_controller_capability",
    "register_controller_master",
]
