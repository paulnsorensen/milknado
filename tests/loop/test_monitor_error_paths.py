from __future__ import annotations

import os
import signal
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import NoReturn

import psutil
import pytest

import milknado.loop._process_observation as observation
from milknado.adapters._loop_worker_evidence import LoopWorkerEvidence
from milknado.domains.common import WorkerIdentity, WorkerOwner
from milknado.domains.graph import MikadoGraph, WorkerEvidenceStore
from milknado.loop._process_contract import ProtectionContext
from milknado.loop._process_gate import SpawnOptions, WorkerProcess
from milknado.loop._process_lifecycle import spawn_protected


@pytest.mark.skipif(os.name == "nt", reason="POSIX process groups required")
def test_unknown_identity_before_observation_does_not_claim_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    node = graph.add_node("worker")
    graph.runs.start("run-1", node.id, "worker.log", "2026-01-01T00:00:00+00:00", None)
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True
    )
    supervisor = psutil.Process()
    identity = WorkerIdentity(
        "inv-1", process.pid, process.pid, psutil.Process(process.pid).create_time()
    )
    try:
        graph.runs.record_worker(
            WorkerOwner("run-1", supervisor.pid, supervisor.create_time(), "run-1", node.id),
            identity,
        )

        def unknown_identity(_pid: int, _token: float) -> str:
            return "unknown"

        monkeypatch.setattr(observation, "identity_state", unknown_identity)
        with pytest.raises(RuntimeError, match="before observation"):
            _ = observation.snapshot(
                WorkerProcess(process, identity, None), LoopWorkerEvidence(graph.db_path)
            )
        record = graph.runs.get_worker("inv-1")
        assert record is not None and record.snapshot_seq == 0
        assert record.observation_owner is None
        assert process.poll() is None
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            _ = process.wait(timeout=2)
        graph.close()


@pytest.mark.skipif(os.name == "nt", reason="POSIX process groups required")
def test_identity_becomes_unknown_after_observation_retains_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    node = graph.add_node("worker")
    graph.runs.start("run-1", node.id, "worker.log", "2026-01-01T00:00:00+00:00", None)
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True
    )
    supervisor = psutil.Process()
    identity = WorkerIdentity(
        "inv-1", process.pid, process.pid, psutil.Process(process.pid).create_time()
    )
    try:
        graph.runs.record_worker(
            WorkerOwner("run-1", supervisor.pid, supervisor.create_time(), "run-1", node.id),
            identity,
        )
        states = iter(("live", "unknown"))

        def changing_identity(_pid: int, _token: float) -> str:
            return next(states)

        monkeypatch.setattr(observation, "identity_state", changing_identity)
        with pytest.raises(RuntimeError, match="during observation"):
            _ = observation.snapshot(
                WorkerProcess(process, identity, None), LoopWorkerEvidence(graph.db_path)
            )
        record = graph.runs.get_worker("inv-1")
        assert record is not None and record.snapshot_seq == 0 and record.observation_seq == 1
        assert record.observation_owner == "supervisor"
        assert record.ended_at is None
        assert process.poll() is None
        with pytest.raises(RuntimeError, match="worker evidence unavailable"):
            _ = observation.snapshot(
                WorkerProcess(process, identity, None), LoopWorkerEvidence(graph.db_path)
            )
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            _ = process.wait(timeout=2)
        graph.close()


@pytest.mark.skipif(os.name == "nt", reason="POSIX lifeline requires passed file descriptors")
def test_sqlite_observation_failure_stops_worker_and_keeps_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    node = graph.add_node("worker")
    graph.runs.start("run-1", node.id, "worker.log", "2026-01-01T00:00:00+00:00", None)
    supervisor = psutil.Process()
    owner = WorkerOwner("run-1", supervisor.pid, supervisor.create_time(), "run-1", node.id)
    protected = spawn_protected(
        SpawnOptions(
            (sys.executable, "-c", "import time; time.sleep(30)"),
            tmp_path,
            None,
            True,
            subprocess.DEVNULL,
            subprocess.PIPE,
            subprocess.PIPE,
        ),
        ProtectionContext(LoopWorkerEvidence(graph.db_path), owner, graph.db_path),
    )
    try:

        def fail_commit(_store: WorkerEvidenceStore, _key: object, _targets: object) -> NoReturn:
            raise sqlite3.OperationalError("observation locked")

        monkeypatch.setattr(WorkerEvidenceStore, "commit", fail_commit)
        deadline = time.monotonic() + 5
        while protected.process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        assert protected.process.poll() is not None
        record = graph.runs.get_worker(protected.identity.invocation_id)
        assert record is not None and record.observation_owner == "supervisor"
        assert record.ended_at is None
    finally:
        protected.close_lifeline()
        if protected.process.poll() is None:
            os.killpg(protected.process.pid, signal.SIGKILL)
        _ = protected.process.wait(timeout=2)
        helper = protected._helper  # pyright: ignore[reportPrivateUsage]
        if helper.poll() is None:
            helper.kill()
        _ = helper.wait(timeout=2)
        graph.close()
