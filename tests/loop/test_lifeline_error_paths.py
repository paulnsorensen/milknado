from __future__ import annotations

import os
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil
import pytest

from milknado.domains.common import HelperIdentity, WorkerIdentity, WorkerOwner
from milknado.domains.graph import MikadoGraph, WorkerEvidenceStore
from milknado.loop._lifeline import _cleanup  # pyright: ignore[reportPrivateUsage]


@pytest.mark.skipif(os.name == "nt", reason="POSIX process groups required")
def test_lifeline_read_failure_retains_live_worker_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    node = graph.add_node("worker")
    graph.runs.start("run-1", node.id, "worker.log", "2026-01-01T00:00:00+00:00", None)
    worker = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True
    )
    supervisor = psutil.Process()
    helper = HelperIdentity("inv-1", 0, supervisor.pid, supervisor.create_time())
    try:
        graph.runs.record_worker(
            WorkerOwner("run-1", supervisor.pid, supervisor.create_time(), "run-1", node.id),
            WorkerIdentity(
                "inv-1", worker.pid, worker.pid, psutil.Process(worker.pid).create_time()
            ),
        )
        graph.runs.record_helper(helper)

        def fail_read(_store: WorkerEvidenceStore, _invocation_id: str) -> None:
            raise sqlite3.OperationalError("evidence locked")

        monkeypatch.setattr(WorkerEvidenceStore, "get", fail_read)
        with WorkerEvidenceStore(graph.db_path) as store:
            assert _cleanup(store, helper, time.monotonic() + 1) == 1
        assert worker.poll() is None
    finally:
        if worker.poll() is None:
            os.killpg(worker.pid, signal.SIGKILL)
            _ = worker.wait(timeout=2)
        graph.close()


@pytest.mark.skipif(os.name == "nt", reason="POSIX process groups required")
def test_lifeline_completion_failure_keeps_open_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    node = graph.add_node("worker")
    graph.runs.start("run-1", node.id, "worker.log", "2026-01-01T00:00:00+00:00", None)
    worker = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True
    )
    supervisor = psutil.Process()
    helper = HelperIdentity("inv-1", 0, supervisor.pid, supervisor.create_time())
    try:
        graph.runs.record_worker(
            WorkerOwner("run-1", supervisor.pid, supervisor.create_time(), "run-1", node.id),
            WorkerIdentity(
                "inv-1", worker.pid, worker.pid, psutil.Process(worker.pid).create_time()
            ),
        )
        graph.runs.record_helper(helper)

        end_attempted = threading.Event()

        def fail_end(
            _store: WorkerEvidenceStore, _invocation_id: str, _sequence: int, _generation: int
        ) -> None:
            end_attempted.set()
            raise sqlite3.OperationalError("completion locked")

        monkeypatch.setattr(WorkerEvidenceStore, "end", fail_end)
        reaper = threading.Thread(target=worker.wait, kwargs={"timeout": 3})
        reaper.start()
        with WorkerEvidenceStore(graph.db_path) as store:
            assert _cleanup(store, helper, time.monotonic() + 3) == 1
        reaper.join(timeout=2)
        assert not reaper.is_alive()
        assert end_attempted.is_set()
        assert worker.returncode is not None and worker.returncode != 0
        record = graph.runs.get_worker("inv-1")
        assert record is not None and record.ended_at is None
        assert record.observation_owner is None
    finally:
        if worker.poll() is None:
            os.killpg(worker.pid, signal.SIGKILL)
            _ = worker.wait(timeout=2)
        graph.close()


@pytest.mark.parametrize(
    "arguments",
    [
        (),
        ("graph.db", "not-an-fd", "inv-1", "0"),
        ("graph.db", "0", "inv-1", "not-a-generation"),
        ("graph.db", "-1", "inv-1", "0"),
        ("graph.db", "0", "inv-1", "-1"),
        ("graph.db", "0", "", "0"),
    ],
)
def test_lifeline_bootstrap_rejects_invalid_arguments(arguments: tuple[str, ...]) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "milknado.adapters._loop_lifeline", *arguments],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    assert result.returncode == 2
    assert "Traceback" not in result.stderr


@pytest.mark.skipif(os.name == "nt", reason="POSIX process groups required")
def test_lifeline_failed_observation_keeps_worker_and_open_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    node = graph.add_node("worker")
    graph.runs.start("run-1", node.id, "worker.log", "2026-01-01T00:00:00+00:00", None)
    worker = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True
    )
    supervisor = psutil.Process()
    helper = HelperIdentity("inv-1", 0, supervisor.pid, supervisor.create_time())
    try:
        graph.runs.record_worker(
            WorkerOwner("run-1", supervisor.pid, supervisor.create_time(), "run-1", node.id),
            WorkerIdentity(
                "inv-1", worker.pid, worker.pid, psutil.Process(worker.pid).create_time()
            ),
        )
        graph.runs.record_helper(helper)
        commit_attempted = False

        def fail_commit(_store: WorkerEvidenceStore, _key: object, _descendants: object) -> None:
            nonlocal commit_attempted
            commit_attempted = True
            raise sqlite3.OperationalError("observation locked")

        monkeypatch.setattr(WorkerEvidenceStore, "commit", fail_commit)
        with WorkerEvidenceStore(graph.db_path) as store:
            assert _cleanup(store, helper, time.monotonic() + 1) == 1
        assert commit_attempted
        assert worker.poll() is None
        record = graph.runs.get_worker("inv-1")
        assert record is not None and record.observation_owner == "helper"
        assert record.ended_at is None
    finally:
        if worker.poll() is None:
            os.killpg(worker.pid, signal.SIGKILL)
            _ = worker.wait(timeout=2)
        graph.close()
