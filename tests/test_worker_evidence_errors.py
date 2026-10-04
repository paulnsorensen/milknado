from __future__ import annotations

import os
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
from threading import Barrier, Lock

import pytest

import milknado.domains.graph.worker_evidence as evidence_module
from milknado.domains.common import WorkerIdentity, WorkerOwner
from milknado.domains.graph import (
    MikadoGraph,
    UnassociatedWorkers,
    WorkerEvidenceStore,
    existing_standalone_worker_db,
    open_standalone_worker_evidence,
)

pytestmark = pytest.mark.skipif(os.name == "nt", reason="POSIX private worker store")


def test_concurrent_first_open_creates_missing_private_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = tmp_path / "fresh"
    path = parent / "workers.db"
    both_ready = Barrier(2)
    count_lock = Lock()
    calls = 0
    original_mkdir = Path.mkdir

    def coordinated_mkdir(
        self: Path, mode: int = 0o777, parents: bool = False, exist_ok: bool = False
    ) -> None:
        nonlocal calls
        wait = False
        if self == parent:
            with count_lock:
                calls += 1
                wait = calls <= 2
        if wait:
            _ = both_ready.wait(timeout=2)
        original_mkdir(self, mode=mode, parents=parents, exist_ok=exist_ok)

    monkeypatch.setattr(Path, "mkdir", coordinated_mkdir)

    def open_store() -> None:
        with open_standalone_worker_evidence(path) as store:
            assert store.db_path == path

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = [pool.submit(open_store) for _ in range(2)]
        for result in results:
            result.result(timeout=5)
    assert parent.stat().st_mode & 0o077 == 0
    assert path.stat().st_mode & 0o077 == 0


def test_unsafe_parent_refuses_store_creation(tmp_path: Path) -> None:
    parent = tmp_path / "insecure"
    parent.mkdir(mode=0o755)
    parent.chmod(0o755)
    path = parent / "workers.db"

    with pytest.raises(RuntimeError, match="not private"):
        _ = open_standalone_worker_evidence(path)
    assert not path.exists()


def test_symlink_parent_refuses_store_creation(tmp_path: Path) -> None:
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    alias = tmp_path / "alias"
    alias.symlink_to(private, target_is_directory=True)
    path = alias / "workers.db"

    with pytest.raises(RuntimeError, match="symlink"):
        _ = open_standalone_worker_evidence(path)
    assert not (private / "workers.db").exists()


def test_current_version_with_missing_worker_column_is_not_recreated(tmp_path: Path) -> None:
    path = tmp_path / "graph.db"
    graph = MikadoGraph(path)
    graph.close()
    path.chmod(0o600)
    with closing(sqlite3.connect(path)) as conn, conn:
        _ = conn.execute("DROP TABLE run_workers")
        _ = conn.execute("CREATE TABLE run_workers (invocation_id TEXT)")

    with pytest.raises(RuntimeError, match="schema is not current"):
        _ = WorkerEvidenceStore(path)
    with closing(sqlite3.connect(path)) as conn:
        columns = conn.execute("PRAGMA table_info(run_workers)").fetchall()
    assert len(columns) == 1
    assert existing_standalone_worker_db(path)


def test_expired_store_deadline_refuses_evidence_operation(tmp_path: Path) -> None:
    path = tmp_path / "workers.db"
    with open_standalone_worker_evidence(path) as store:
        store.set_deadline(time.monotonic() - 1)
        with pytest.raises(TimeoutError, match="deadline expired"):
            _ = store.live_workers(UnassociatedWorkers())
    with open_standalone_worker_evidence(path) as store:
        assert store.live_workers(UnassociatedWorkers()) == ()


def test_repeated_open_keeps_worker_lock_private(tmp_path: Path) -> None:
    path = tmp_path / "workers.db"
    lock_path = path.with_suffix(path.suffix + ".lock")
    for _ in range(2):
        with open_standalone_worker_evidence(path):
            assert lock_path.stat().st_mode & 0o077 == 0


def _capture_initialization_connection(
    path: Path, monkeypatch: pytest.MonkeyPatch
) -> list[sqlite3.Connection]:
    original_connect = sqlite3.connect
    opened: list[sqlite3.Connection] = []

    def connect(
        database: str | Path, *, timeout: float = 5.0, uri: bool = False
    ) -> sqlite3.Connection:
        conn = original_connect(database, timeout=timeout, uri=uri)
        if database == path:
            opened.append(conn)
        return conn

    monkeypatch.setattr(sqlite3, "connect", connect)
    return opened


def test_initialization_connection_closes_after_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "workers.db"
    opened = _capture_initialization_connection(path, monkeypatch)
    with open_standalone_worker_evidence(path) as store:
        assert store.live_workers(UnassociatedWorkers()) == ()
    assert len(opened) == 1
    try:
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            _ = opened[0].execute("SELECT 1")
    finally:
        opened[0].close()


def test_initialization_failure_closes_connection_and_removes_new_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "workers.db"
    opened = _capture_initialization_connection(path, monkeypatch)

    def fail_schema(_conn: sqlite3.Connection) -> None:
        raise RuntimeError("schema fault")

    monkeypatch.setattr(evidence_module, "create_tables", fail_schema)
    with pytest.raises(RuntimeError, match="schema fault"):
        _ = open_standalone_worker_evidence(path)
    assert not path.exists()
    assert len(opened) == 1
    try:
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            _ = opened[0].execute("SELECT 1")
    finally:
        opened[0].close()


@pytest.mark.parametrize(
    "owner",
    [
        WorkerOwner("runtime-1", 999999, 123.5, "graph-run", None),
        WorkerOwner("runtime-1", 999999, 123.5, None, 1),
        WorkerOwner("", 999999, 123.5),
        WorkerOwner("runtime-1", 0, 123.5),
        WorkerOwner("runtime-1", 999999, 0),
    ],
)
def test_invalid_worker_owner_never_writes_evidence(tmp_path: Path, owner: WorkerOwner) -> None:
    with open_standalone_worker_evidence(tmp_path / "workers.db") as store:
        with pytest.raises(ValueError):
            store.record_worker(owner, WorkerIdentity("inv-1", 2345, 2345, 123.5))
        assert store.live_workers(UnassociatedWorkers()) == ()
