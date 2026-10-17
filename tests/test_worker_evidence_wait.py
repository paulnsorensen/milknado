"""Worker evidence writes wait out a commit stall longer than one second."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from threading import Event, Thread

import pytest
from filelock import FileLock

from milknado.domains.common import ObservationKey, WorkerIdentity, WorkerOwner
from milknado.domains.graph import MikadoGraph, WorkerEvidenceStore
from milknado.domains.graph.worker_evidence import open_standalone_worker_evidence

_COMMIT_STALL_S = 1.5


def _graph_with_worker(tmp_path: Path) -> Path:
    graph = MikadoGraph(tmp_path / "graph.db")
    node = graph.add_node("worker")
    graph.runs.start("run-1", node.id, "worker.log", "2026-01-01T00:00:00+00:00", None)
    graph.runs.record_worker(
        WorkerOwner("run-1", 999999, 123.5, "run-1", node.id),
        WorkerIdentity("inv-1", 2345, 2345, 123.5),
    )
    graph.close()
    return tmp_path / "graph.db"


def _stall_commit(db_path: Path, locked: Event) -> None:
    writer = sqlite3.connect(db_path)
    try:
        _ = writer.execute("BEGIN IMMEDIATE")
        locked.set()
        time.sleep(_COMMIT_STALL_S)
        writer.commit()
    finally:
        writer.close()


@pytest.mark.parametrize("deadline_s", [3.0, None], ids=["deadline", "no-deadline"])
def test_write_outlasts_a_commit_stall_longer_than_one_second(
    tmp_path: Path, deadline_s: float | None
) -> None:
    db_path = _graph_with_worker(tmp_path)
    locked = Event()
    stall = Thread(target=_stall_commit, args=(db_path, locked))
    stall.start()
    try:
        assert locked.wait(5)
        start = time.monotonic()
        deadline = None if deadline_s is None else start + deadline_s
        with WorkerEvidenceStore(db_path, deadline=deadline) as store:
            store.begin(ObservationKey("inv-1", "supervisor", 1, -1, 2345, 123.5))
        assert time.monotonic() - start >= 1.0
    finally:
        stall.join(10)


def _hold_open_lock(lock_path: Path, held: Event) -> None:
    with FileLock(lock_path, mode=0o600):
        held.set()
        time.sleep(_COMMIT_STALL_S)


def test_first_open_outlasts_a_creation_lock_held_longer_than_one_second(
    tmp_path: Path,
) -> None:
    path = tmp_path / "workers.db"
    held = Event()
    holder = Thread(target=_hold_open_lock, args=(tmp_path / "workers.db.lock", held))
    holder.start()
    try:
        assert held.wait(5)
        start = time.monotonic()
        with open_standalone_worker_evidence(path):
            pass
        assert time.monotonic() - start >= 1.0
    finally:
        holder.join(10)
