from __future__ import annotations

import os
import select
import signal
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import psutil
import pytest

from milknado.domains.common import HelperIdentity, WorkerIdentity
from milknado.domains.graph import MikadoGraph


@pytest.mark.skipif(os.name == "nt", reason="POSIX lifeline requires passed file descriptors")
def test_lifeline_ready_then_supervisor_eof_stops_worker(tmp_path: Path) -> None:
    db_path = tmp_path / "graph.db"
    graph = MikadoGraph(db_path)
    node = graph.add_node("worker")
    graph.runs.start("run-1", node.id, "worker.log", "2026-01-01T00:00:00+00:00", None)
    worker = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True
    )
    lifeline_read, lifeline_write = os.pipe()
    helper = None
    try:
        identity = WorkerIdentity(
            "inv-1", worker.pid, worker.pid, psutil.Process(worker.pid).create_time()
        )
        graph.runs.record_worker("run-1", identity)
        helper = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "milknado.adapters._loop_lifeline",
                str(db_path),
                str(lifeline_read),
                "inv-1",
                "0",
            ],
            pass_fds=(lifeline_read,),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        os.close(lifeline_read)
        lifeline_read = -1
        graph.runs.record_helper(
            HelperIdentity("inv-1", 0, helper.pid, psutil.Process(helper.pid).create_time())
        )
        assert helper.stdout is not None
        ready, _, _ = select.select([helper.stdout], [], [], 8)
        assert ready
        assert helper.stdout.readline().startswith("READY ")
        os.close(lifeline_write)
        lifeline_write = -1
        assert worker.wait(timeout=5) != 0
        assert helper.wait(timeout=5) == 0
        assert graph.runs.live_workers(run_id="run-1") == ()
    finally:
        if lifeline_read != -1:
            os.close(lifeline_read)
        if lifeline_write != -1:
            os.close(lifeline_write)
        if helper is not None and helper.poll() is None:
            helper.kill()
            helper.wait(timeout=1)
        if worker.poll() is None:
            os.killpg(worker.pid, signal.SIGKILL)
            worker.wait(timeout=1)
        graph.close()


@pytest.mark.skipif(os.name == "nt", reason="POSIX lifeline requires passed file descriptors")
def test_lifeline_eof_does_not_extend_cleanup_for_busy_database(tmp_path: Path) -> None:
    db_path = tmp_path / "graph.db"
    graph = MikadoGraph(db_path)
    node = graph.add_node("worker")
    graph.runs.start("run-1", node.id, "worker.log", "2026-01-01T00:00:00+00:00", None)
    worker = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True
    )
    read_fd, write_fd = os.pipe()
    helper = None
    lock = sqlite3.connect(db_path)
    try:
        graph.runs.record_worker(
            "run-1", WorkerIdentity("inv-1", worker.pid, worker.pid, psutil.Process(worker.pid).create_time())
        )
        helper = subprocess.Popen(
            [sys.executable, "-m", "milknado.adapters._loop_lifeline", str(db_path),
             str(read_fd), "inv-1", "0"],
            pass_fds=(read_fd,), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        os.close(read_fd)
        read_fd = -1
        graph.runs.record_helper(
            HelperIdentity("inv-1", 0, helper.pid, psutil.Process(helper.pid).create_time())
        )
        assert helper.stdout is not None
        ready, _, _ = select.select([helper.stdout], [], [], 8)
        assert ready
        assert helper.stdout.readline().startswith("READY ")
        lock.execute("BEGIN IMMEDIATE")
        lock.execute("UPDATE run_workers SET snapshot_seq = snapshot_seq WHERE invocation_id = ?", ("inv-1",))
        start = time.monotonic()
        os.close(write_fd)
        write_fd = -1
        assert helper.wait(timeout=4.5) != 0
        assert time.monotonic() - start < 4.0
        assert helper.stderr is not None
        assert "Traceback" not in helper.stderr.read()
        assert worker.poll() is None
        lock.rollback()
        assert len(graph.runs.live_workers(run_id="run-1")) == 1
    finally:
        lock.rollback()
        lock.close()
        if read_fd != -1:
            os.close(read_fd)
        if write_fd != -1:
            os.close(write_fd)
        if helper is not None and helper.poll() is None:
            helper.kill()
            helper.wait(timeout=1)
        if worker.poll() is None:
            os.killpg(worker.pid, signal.SIGKILL)
            worker.wait(timeout=1)
        graph.close()
