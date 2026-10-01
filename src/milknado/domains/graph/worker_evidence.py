"""Current-schema worker evidence access for recovery and lifelines."""

from __future__ import annotations

import os
import sqlite3
import stat
import time
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Self, cast

from filelock import FileLock

import milknado.domains.graph._worker_persistence as _worker_persistence
from milknado.domains.common import HelperIdentity, ObservationKey, WorkerIdentity, WorkerOwner
from milknado.domains.graph._persistence import SCHEMA_VERSION, create_tables, migrate
from milknado.domains.graph._sqlite_rows import fetchall, fetchone


@dataclass(frozen=True, slots=True)
class NodeWorkers:
    node_id: int


@dataclass(frozen=True, slots=True)
class RunWorkers:
    graph_run_id: str


@dataclass(frozen=True, slots=True)
class UnassociatedWorkers:
    pass


WorkerSelection = NodeWorkers | RunWorkers | UnassociatedWorkers


def default_worker_db_path() -> Path:
    state = os.environ.get("XDG_STATE_HOME", "").strip()
    root = Path(state) if state else Path.home() / ".local" / "state"
    if not root.is_absolute():
        raise ValueError("XDG_STATE_HOME must be absolute")
    return root / "milknado" / "loop-workers.sqlite3"


def _check_private(path: Path, *, directory: bool) -> None:
    details = path.lstat()
    if stat.S_ISLNK(details.st_mode):
        raise RuntimeError(f"worker evidence path is a symlink: {path}")
    if details.st_uid != os.getuid():
        raise RuntimeError(f"worker evidence path has foreign owner: {path}")
    if details.st_mode & 0o077:
        raise RuntimeError(f"worker evidence path is not private: {path}")
    if directory and not stat.S_ISDIR(details.st_mode):
        raise RuntimeError(f"worker evidence path is not a directory: {path}")
    if not directory and not stat.S_ISREG(details.st_mode):
        raise RuntimeError(f"worker evidence path is not a file: {path}")


def existing_standalone_worker_db(path: Path) -> bool:
    """Validate an existing standalone store without creating one."""
    if not path.is_absolute():
        raise ValueError("worker evidence path must be absolute")
    try:
        _check_private(path, directory=False)
    except FileNotFoundError:
        return False
    _check_private(path.parent, directory=True)
    return True


def open_standalone_worker_evidence(
    db_path: Path | None = None,
    *,
    deadline: float | None = None,
) -> WorkerEvidenceStore:
    path = default_worker_db_path() if db_path is None else db_path
    if not path.is_absolute():
        raise ValueError("worker evidence path must be absolute")
    parent = path.parent
    if not parent.exists():
        parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    _check_private(parent, directory=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(descriptor)
    except FileExistsError:
        _check_private(lock_path, directory=False)
    timeout = 1 if deadline is None else max(0, deadline - time.monotonic())
    with FileLock(lock_path, timeout=timeout, mode=0o600):
        if not path.exists():
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            os.close(descriptor)
            try:
                with closing(sqlite3.connect(path)) as conn, conn:
                    create_tables(conn)
                    migrate(conn)
            except Exception:
                path.unlink()
                raise
        _check_private(path, directory=False)
        return WorkerEvidenceStore(path, deadline=deadline)


class WorkerEvidenceStore:
    def __init__(self, db_path: Path, *, deadline: float | None = None) -> None:
        if not db_path.is_file():
            raise FileNotFoundError(db_path)
        remaining = 1 if deadline is None else deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("worker evidence deadline expired")
        uri = f"{db_path.resolve().as_uri()}?mode=rw"
        conn = sqlite3.connect(uri, uri=True, timeout=min(1, remaining))
        self._conn: sqlite3.Connection = conn
        self._db_path: Path = db_path
        self._deadline: float | None = deadline
        try:
            conn.row_factory = sqlite3.Row
            self._limit_wait()
            row = fetchone(conn, "PRAGMA user_version")
            columns: set[str] = set()
            for item in fetchall(conn, "PRAGMA table_info(run_workers)"):
                name = cast(object, item[1])
                if not isinstance(name, str):
                    raise RuntimeError("invalid worker evidence schema")
                columns.add(name)
            if (
                row is None
                or cast(int, row[0]) != SCHEMA_VERSION
                or "runtime_run_id" not in columns
            ):
                raise RuntimeError("worker evidence schema is not current")
        except Exception:
            conn.close()
            raise

    @property
    def db_path(self) -> Path:
        return self._db_path

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        self._conn.close()

    def set_deadline(self, deadline: float) -> None:
        self._deadline = deadline

    def _limit_wait(self) -> None:
        if self._deadline is None:
            return
        remaining_ms = int((self._deadline - time.monotonic()) * 1000)
        if remaining_ms <= 0:
            raise TimeoutError("worker evidence deadline expired")
        _ = self._conn.execute(f"PRAGMA busy_timeout={min(1000, remaining_ms)}")

    def record_worker(self, owner: WorkerOwner, worker: WorkerIdentity) -> None:
        self._limit_wait()
        _worker_persistence.record_worker(self._conn, owner, worker)

    def record_helper(self, helper: HelperIdentity) -> None:
        self._limit_wait()
        _worker_persistence.record_helper(self._conn, helper)

    def live_workers(
        self, selection: WorkerSelection
    ) -> tuple[_worker_persistence.WorkerRecord, ...]:
        self._limit_wait()
        if isinstance(selection, NodeWorkers):
            return _worker_persistence.live_workers(self._conn, node_id=selection.node_id)
        if isinstance(selection, RunWorkers):
            return _worker_persistence.live_workers(self._conn, run_id=selection.graph_run_id)
        if isinstance(cast(object, selection), UnassociatedWorkers):
            return _worker_persistence.live_workers(self._conn, unassociated=True)
        raise TypeError("worker selection required")

    def get(self, invocation_id: str) -> _worker_persistence.WorkerRecord | None:
        self._limit_wait()
        return _worker_persistence.get_worker(self._conn, invocation_id)

    def begin(self, key: ObservationKey) -> None:
        self._limit_wait()
        _worker_persistence.begin_observation(self._conn, key)

    def commit(
        self, key: ObservationKey, descendants: tuple[_worker_persistence.Descendant, ...]
    ) -> None:
        self._limit_wait()
        _worker_persistence.commit_observation(self._conn, key, descendants)

    def ready(self, helper: HelperIdentity, sequence: int) -> bool:
        self._limit_wait()
        return _worker_persistence.ready_helper(self._conn, helper, sequence)

    def end(
        self, invocation_id: str, snapshot_seq: int, helper_generation: int | None = None
    ) -> None:
        self._limit_wait()
        _worker_persistence.end_worker(self._conn, invocation_id, snapshot_seq, helper_generation)
