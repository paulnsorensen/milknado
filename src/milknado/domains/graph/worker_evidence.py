"""Existing-schema worker evidence access for a lifeline process."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Self, cast

from milknado.domains.common import HelperIdentity, ObservationKey
import milknado.domains.graph._worker_persistence as _worker_persistence
from milknado.domains.graph._persistence import SCHEMA_VERSION
from milknado.domains.graph._sqlite_rows import fetchone


class WorkerEvidenceStore:
    def __init__(self, db_path: Path) -> None:
        if not db_path.is_file():
            raise FileNotFoundError(db_path)
        uri = f"{db_path.resolve().as_uri()}?mode=rw"
        conn = sqlite3.connect(uri, uri=True)
        conn.row_factory = sqlite3.Row
        _ = conn.execute("PRAGMA busy_timeout=1000")
        row = fetchone(conn, "PRAGMA user_version")
        if row is None or cast(int, row[0]) != SCHEMA_VERSION:
            conn.close()
            raise RuntimeError("worker evidence schema is not current")
        self._conn = conn
        self._deadline: float | None = None

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

    def end(self, invocation_id: str) -> None:
        self._limit_wait()
        _worker_persistence.end_worker(self._conn, invocation_id)
