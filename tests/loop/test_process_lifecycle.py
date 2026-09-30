from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import psutil
import pytest

from milknado.domains.common import WorkerIdentity
import milknado.loop._process_identity as process_identity
from milknado.loop._process_lifecycle import SpawnOptions, spawn_gated, terminate_verified


@pytest.mark.skipif(os.name == "nt", reason="POSIX group identity is required")
def test_verified_worker_group_stops_real_child(tmp_path: Path) -> None:
    marker = tmp_path / "child.pid"
    script = (
        "import subprocess,sys,time;"
        "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']);"
        "open(sys.argv[1],'w').write(str(child.pid));"
        "time.sleep(30)"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", script, str(marker)], start_new_session=True
    )
    try:
        deadline = time.monotonic() + 3
        child_pid = 0
        while time.monotonic() < deadline:
            if marker.exists() and (pid_text := marker.read_text()):
                child_pid = int(pid_text)
                break
            time.sleep(0.01)
        assert child_pid > 0
        identity = WorkerIdentity("inv-1", proc.pid, proc.pid, psutil.Process(proc.pid).create_time())
        result = terminate_verified(identity, (), time.monotonic() + 3, proc)
        assert result.covered_exited is True
        assert proc.wait(timeout=1) != 0
        assert not psutil.pid_exists(child_pid) or psutil.Process(child_pid).status() == psutil.STATUS_ZOMBIE
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=1)


@pytest.mark.skipif(os.name == "nt", reason="POSIX group identity is required")
def test_mismatched_worker_token_refuses_signal() -> None:
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True
    )
    try:
        identity = WorkerIdentity(
            "inv-1", proc.pid, proc.pid, psutil.Process(proc.pid).create_time() + 100
        )
        result = terminate_verified(identity, (), time.monotonic() + 0.2)
        assert result.covered_exited is False
        assert proc.poll() is None
    finally:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=1)


@pytest.mark.skipif(os.name == "nt", reason="POSIX exec gate is required")
def test_exec_gate_waits_for_explicit_release(tmp_path: Path) -> None:
    marker = tmp_path / "worked"
    worker = spawn_gated(
        SpawnOptions(
            (sys.executable, "-c", f"from pathlib import Path; Path({str(marker)!r}).touch()"),
            tmp_path,
            None,
            True,
            subprocess.DEVNULL,
            subprocess.PIPE,
            subprocess.PIPE,
        )
    )
    try:
        assert worker.process.pid == worker.identity.pid
        assert not marker.exists()
        worker.release()
        assert worker.process.wait(timeout=3) == 0
        assert marker.exists()
    finally:
        worker.close_gate()


@pytest.mark.skipif(os.name == "nt", reason="POSIX exec gate is required")
def test_exec_gate_eof_aborts_without_agent_work(tmp_path: Path) -> None:
    marker = tmp_path / "worked"
    worker = spawn_gated(
        SpawnOptions(
            (sys.executable, "-c", f"from pathlib import Path; Path({str(marker)!r}).touch()"),
            tmp_path,
            None,
            True,
            subprocess.DEVNULL,
            subprocess.PIPE,
            subprocess.PIPE,
        )
    )
    worker.close_gate()
    assert worker.process.wait(timeout=3) != 0
    assert not marker.exists()


@pytest.mark.skipif(os.name == "nt", reason="POSIX signal deadline is required")
def test_verified_cleanup_never_signals_after_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signals: list[int] = []

    def slow_identity(_pid: int, _token: float) -> str:
        time.sleep(0.03)
        return "live"

    monkeypatch.setattr(process_identity, "identity_state", slow_identity)
    monkeypatch.setattr(process_identity.os, "kill", lambda _pid, sig: signals.append(sig))
    monkeypatch.setattr(process_identity.os, "killpg", lambda _pgid, sig: signals.append(sig))
    monkeypatch.setattr(process_identity, "_group_state", lambda _pgid: "gone")
    worker = WorkerIdentity("inv-1", 2345, 2345, 123.5)
    result = terminate_verified(worker, (), time.monotonic() + 0.01)
    assert signals == []
    assert result.covered_exited is False
