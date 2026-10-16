"""Reproduce a frozen SQLite writer during a committed worker gap."""

from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
import threading
import time
from contextlib import closing
from pathlib import Path

import psutil
import pytest

from milknado.domains.common import ObservationKey, WorkerIdentity, WorkerOwner
from milknado.domains.graph import MikadoGraph
from tests.test_loop_recovery_subprocess import freeze_after_snapshot

if os.name != "posix":
    pytest.skip("POSIX process signals are required", allow_module_level=True)

_DEADLINE = 8
_LOCK_HOLDER = """\
import sqlite3
import sys
import time
from pathlib import Path

db, go, ready, release, released = map(Path, sys.argv[1:])
while not go.exists():
    time.sleep(0.01)
conn = sqlite3.connect(db)
conn.execute("BEGIN IMMEDIATE")
ready.write_text("locked")
while not release.exists():
    time.sleep(0.01)
released.write_text("resumed")
conn.rollback()
while True:
    time.sleep(1)
"""


def _wait_for_file(path: Path) -> None:
    deadline = time.monotonic() + _DEADLINE
    while not path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert path.exists(), f"{path.name} was never written"


def _release_after_busy_probe(db: Path, pid: int, release: Path, errors: list[str]) -> None:
    """Release the child only after a stopped child blocks a separate writer."""
    deadline = time.monotonic() + _DEADLINE
    while psutil.Process(pid).status() != psutil.STATUS_STOPPED:
        if time.monotonic() >= deadline:
            errors.append("child was never stopped")
            return
        time.sleep(0.01)
    try:
        with closing(sqlite3.connect(db, timeout=0)) as probe:
            _ = probe.execute("BEGIN IMMEDIATE")
            probe.rollback()
    except sqlite3.OperationalError as exc:
        if exc.sqlite_errorcode != sqlite3.SQLITE_BUSY:
            errors.append(f"unexpected SQLite error: {exc}")
            return
        _ = release.write_text("release")
        return
    errors.append("stopped child did not hold the writer lock")


def test_frozen_writer_resumes_before_observation_gap_is_accepted(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    names = ("go", "ready", "release", "released")
    go, ready, release, released = (tmp_path / name for name in names)
    argv = [sys.executable, "-c", _LOCK_HOLDER, str(graph.db_path)]
    process = subprocess.Popen(
        [*argv, str(go), str(ready), str(release), str(released)], start_new_session=True
    )
    errors: list[str] = []
    watcher: threading.Thread | None = None
    try:
        token = psutil.Process(process.pid).create_time()
        worker = WorkerIdentity("inv-1", process.pid, process.pid, token)
        graph.runs.record_worker(WorkerOwner("run-1", process.pid, token), worker)
        first = ObservationKey(worker.invocation_id, "supervisor", 1, 0, process.pid, token)
        graph.runs.begin_worker_observation(first)
        graph.runs.commit_worker_observation(first, ())
        _ = go.write_text("go")
        _wait_for_file(ready)
        watcher = threading.Thread(
            target=_release_after_busy_probe,
            args=(graph.db_path, process.pid, release, errors),
        )
        watcher.start()
        recorded = graph.runs.get_worker(worker.invocation_id)
        assert recorded is not None
        frozen = freeze_after_snapshot(graph, recorded, process.pid)
        watcher.join(timeout=_DEADLINE)
        assert not watcher.is_alive() and errors == []
        assert release.exists() and released.exists(), "child was not resumed after BUSY"
        assert frozen.snapshot_seq == 1 and frozen.observation_owner is None
        assert psutil.Process(process.pid).status() == psutil.STATUS_STOPPED
        second = ObservationKey(worker.invocation_id, "supervisor", 2, 0, process.pid, token)
        graph.runs.begin_worker_observation(second)
    finally:
        process.kill()
        _ = process.wait(timeout=5)
        if watcher is not None:
            watcher.join(timeout=5)
        graph.close()
