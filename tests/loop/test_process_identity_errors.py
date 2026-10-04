from __future__ import annotations

import os
import signal
import time

import psutil
import pytest

from milknado.domains.common import WorkerIdentity
from milknado.loop._process_identity import (
    identity_state,
    observe_descendants,
    terminate_verified_result,
)

pytestmark = pytest.mark.skipif(os.name == "nt", reason="POSIX process identity")


def test_access_denied_identity_is_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    def inaccessible(_pid: int) -> psutil.Process:
        raise psutil.AccessDenied(pid=987654)

    monkeypatch.setattr(psutil, "Process", inaccessible)
    assert identity_state(987654, 123.5) == "unknown"


def test_descendant_enumeration_denial_is_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    process = psutil.Process()
    worker = WorkerIdentity("inv-1", process.pid, os.getpgid(process.pid), process.create_time())

    def inaccessible(_self: psutil.Process, recursive: bool = False) -> list[psutil.Process]:
        assert recursive
        raise psutil.AccessDenied(pid=process.pid)

    monkeypatch.setattr(psutil.Process, "children", inaccessible)
    with pytest.raises(RuntimeError, match="enumeration unresolved"):
        _ = observe_descendants(worker)


def test_mismatched_identity_never_signals_current_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = psutil.Process()
    worker = WorkerIdentity("inv-1", process.pid, process.pid + 1, process.create_time() - 100)
    signals: list[tuple[int, int]] = []

    def record_signal(pid: int, signum: int) -> None:
        signals.append((pid, signum))

    monkeypatch.setattr(os, "kill", record_signal)
    result = terminate_verified_result(worker, (), time.monotonic() + 0.2)
    assert not result.covered_exited
    assert "worker identity mismatch" in result.unresolved
    assert signals == []
    assert process.is_running()


def test_permission_denied_signal_retains_unresolved_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = psutil.Process()
    worker = WorkerIdentity("inv-1", process.pid, process.pid, process.create_time())
    attempts: list[int] = []
    group_attempts: list[tuple[int, int]] = []

    def deny_signal(pid: int, signum: int) -> None:
        assert signum in (signal.SIGTERM, signal.SIGKILL)
        attempts.append(pid)
        raise PermissionError(pid)

    def same_group(pid: int) -> int:
        assert pid == process.pid
        return pid

    def deny_group(pgid: int, signum: int) -> None:
        group_attempts.append((pgid, signum))
        raise PermissionError(pgid)

    monkeypatch.setattr(os, "kill", deny_signal)
    monkeypatch.setattr(os, "getpgid", same_group)
    monkeypatch.setattr(os, "killpg", deny_group)
    result = terminate_verified_result(worker, (), time.monotonic() + 0.2)
    assert attempts
    assert any(signum in (signal.SIGTERM, signal.SIGKILL) for _, signum in group_attempts)
    assert not result.covered_exited
    assert result.unresolved == ("worker cleanup deadline expired",)
    assert process.is_running()


def test_worker_disappears_during_descendant_enumeration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = psutil.Process()
    worker = WorkerIdentity("inv-1", process.pid, os.getpgid(process.pid), process.create_time())

    def disappeared(_self: psutil.Process, recursive: bool = False) -> list[psutil.Process]:
        assert recursive
        raise psutil.NoSuchProcess(pid=process.pid)

    monkeypatch.setattr(psutil.Process, "children", disappeared)
    assert observe_descendants(worker) == ()


def test_child_disappears_during_descendant_sampling(monkeypatch: pytest.MonkeyPatch) -> None:
    process = psutil.Process()
    worker = WorkerIdentity("inv-1", process.pid, os.getpgid(process.pid), process.create_time())

    def children(_self: psutil.Process, recursive: bool = False) -> list[psutil.Process]:
        assert recursive
        return [process]

    monkeypatch.setattr(psutil.Process, "children", children)

    def disappeared(_pid: int) -> int:
        raise ProcessLookupError

    monkeypatch.setattr(os, "getpgid", disappeared)
    assert observe_descendants(worker) == ()


def test_unknown_group_never_confirms_covered_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    worker = WorkerIdentity("inv-1", 2**31 - 1, 2**31 - 1, 123.5)

    def deny_group(_pgid: int, _signum: int) -> None:
        raise PermissionError("group inspection denied")

    monkeypatch.setattr(os, "killpg", deny_group)
    result = terminate_verified_result(worker, (), time.monotonic() + 0.2)
    assert not result.covered_exited
    assert "worker group unknown" in result.unresolved
