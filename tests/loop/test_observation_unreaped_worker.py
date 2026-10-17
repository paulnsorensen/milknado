"""A gone worker that another thread is still reaping must not abort observation."""

from __future__ import annotations

import subprocess
import sys
import threading
import time
from types import SimpleNamespace
from typing import cast

import psutil
import pytest

from milknado.domains.common import WorkerIdentity, identity_state
from milknado.loop._process_contract import WorkerEvidence
from milknado.loop._process_gate import WorkerProcess
from milknado.loop._process_observation import snapshot


def _zombie() -> subprocess.Popen[bytes]:
    process = subprocess.Popen([sys.executable, "-c", "pass"])
    ps = psutil.Process(process.pid)
    deadline = time.monotonic() + 30
    while ps.status() != psutil.STATUS_ZOMBIE and time.monotonic() < deadline:
        time.sleep(0.01)
    return process


def _gone_worker() -> tuple[WorkerProcess, WorkerEvidence]:
    process = _zombie()
    token = psutil.Process(process.pid).create_time()
    identity = WorkerIdentity("inv-1", process.pid, process.pid, token)
    assert identity_state(identity.pid, identity.start_token) == "gone"
    record = SimpleNamespace(snapshot_seq=7, observation_owner=None)
    evidence = cast(WorkerEvidence, SimpleNamespace(get_worker=lambda _id: record))
    return WorkerProcess(process, identity, None), evidence


def test_snapshot_waits_for_concurrent_reaper_of_gone_worker() -> None:
    worker, evidence = _gone_worker()
    lock = worker.process._waitpid_lock  # pyright: ignore[reportPrivateUsage]
    # Popen.wait() in another thread holds this lock; poll() then reports None.
    lock.acquire()
    assert worker.process.poll() is None
    release = threading.Timer(0.3, lock.release)
    release.start()
    try:
        assert snapshot(worker, evidence) == 7
    finally:
        release.join()
        _ = worker.process.wait(timeout=30)


def test_snapshot_still_rejects_unreaped_worker_after_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker, evidence = _gone_worker()
    lock = worker.process._waitpid_lock  # pyright: ignore[reportPrivateUsage]
    lock.acquire()
    monkeypatch.setattr("milknado.loop._process_observation.REAP_WAIT_SECONDS", 0.05)
    try:
        with pytest.raises(RuntimeError, match="unresolved before observation"):
            _ = snapshot(worker, evidence)
    finally:
        lock.release()
        _ = worker.process.wait(timeout=30)
