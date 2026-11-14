"""A gone worker that another thread is still reaping must not abort observation."""

# The stub implements WorkerEvidence structurally; its parameter names must match the protocol.
# pyright: reportUnusedParameter=false

from __future__ import annotations

import subprocess
import sys
import threading
import time
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import cast

import psutil
import pytest

from milknado.domains.common import (
    HelperIdentity,
    ObservationKey,
    WorkerIdentity,
    WorkerOwner,
    identity_state,
)
from milknado.loop._process_contract import WorkerEvidence, WorkerRecordView
from milknado.loop._process_gate import WorkerProcess
from milknado.loop._process_identity import Descendant
from milknado.loop._process_observation import snapshot


@dataclass(frozen=True)
class _Record:
    invocation_id: str = "inv-1"
    snapshot_seq: int = 7
    ready_generation: int = 0
    helper_generation: int = 0
    helper_pid: int | None = None
    helper_start_token: float | None = None
    observation_owner: str | None = None
    descendants: tuple[Descendant, ...] = ()
    ended_at: str | None = None


class _Evidence:
    """Serves one worker record; observation writes are accepted and discarded."""

    def with_deadline(self, deadline: float) -> WorkerEvidence:
        return self

    def held(self) -> AbstractContextManager[None]:
        return nullcontext()

    def record_worker(self, owner: WorkerOwner, worker: WorkerIdentity) -> None:
        pass

    def get_worker(self, invocation_id: str) -> WorkerRecordView | None:
        return _Record(invocation_id=invocation_id)

    def record_helper(self, helper: HelperIdentity) -> None:
        pass

    def begin_worker_observation(self, key: ObservationKey) -> None:
        pass

    def commit_worker_observation(
        self, key: ObservationKey, descendants: tuple[Descendant, ...]
    ) -> None:
        pass

    def end_worker(
        self, invocation_id: str, snapshot_seq: int, helper_generation: int | None = None
    ) -> None:
        pass


def _waitpid_lock(process: object) -> threading.Lock:
    return cast(threading.Lock, getattr(process, "_waitpid_lock"))  # noqa: B009


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
    return WorkerProcess(process, identity, None), _Evidence()


def test_snapshot_waits_for_concurrent_reaper_of_gone_worker() -> None:
    worker, evidence = _gone_worker()
    lock = _waitpid_lock(worker.process)
    # Popen.wait() in another thread holds this lock; poll() then reports None.
    _ = lock.acquire()
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
    lock = _waitpid_lock(worker.process)
    _ = lock.acquire()
    monkeypatch.setattr("milknado.loop._process_observation.REAP_WAIT_SECONDS", 0.05)
    try:
        with pytest.raises(RuntimeError, match="unresolved before observation"):
            _ = snapshot(worker, evidence)
    finally:
        lock.release()
        _ = worker.process.wait(timeout=30)
