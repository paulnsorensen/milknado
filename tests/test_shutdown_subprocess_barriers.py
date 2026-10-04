from __future__ import annotations

import os
import signal
import sqlite3
import sys
import time
from pathlib import Path

import pytest

from tests.test_shutdown_subprocess import (
    _output,  # pyright: ignore[reportPrivateUsage]
    _owned_cleanup,  # pyright: ignore[reportPrivateUsage]
    _project,  # pyright: ignore[reportPrivateUsage]
    _start,  # pyright: ignore[reportPrivateUsage]
    _wait_exit,  # pyright: ignore[reportPrivateUsage]
    _wait_for,  # pyright: ignore[reportPrivateUsage]
)

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals and PTY required")
_SIGHUP = getattr(signal, "SIGHUP", signal.SIGTERM)

_DISPATCH_BARRIER = """
import os
import time
from pathlib import Path
from milknado.app import _shutdown
from milknado.domains.execution import RunLoop

method = os.environ["SHUTDOWN_DISPATCH_METHOD"]
original = getattr(RunLoop, method)
original_record = _shutdown.ShutdownIntent.record
original_bounded_stop = _shutdown.bounded_stop
resumed = Path(os.environ["SHUTDOWN_DISPATCH_RESUMED"])

def record(self, signum, frame):
    original_record(self, signum, frame)
    Path(os.environ["SHUTDOWN_SIGNAL_RECORDED"]).touch()

def stop_after_admission(stop, deadline):
    confirmed = original_bounded_stop(stop, deadline)
    while (
        not os.getenv("SHUTDOWN_BLOCKED_EXIT")
        and not resumed.exists()
        and time.monotonic() < deadline
    ):
        time.sleep(0.02)
    return confirmed

def blocked(self, *args, **kwargs):
    Path(os.environ["SHUTDOWN_DISPATCH_MARKER"]).touch()
    deadline = time.monotonic() + 20
    while (
        not Path(os.environ["SHUTDOWN_DISPATCH_RELEASE"]).exists()
        and time.monotonic() < deadline
    ):
        time.sleep(0.02)
    result = original(self, *args, **kwargs)
    resumed.touch()
    return result

if method == "_dispatch_if_scheduling_open":
    original_batch = RunLoop._dispatch_batch

    def marked_batch(self, *args, **kwargs):
        Path(os.environ["SHUTDOWN_DISPATCH_ADMITTED"]).touch()
        return original_batch(self, *args, **kwargs)

    RunLoop._dispatch_batch = marked_batch

_shutdown.ShutdownIntent.record = record
_shutdown.bounded_stop = stop_after_admission
setattr(RunLoop, method, blocked)
"""


@pytest.mark.parametrize(
    "mode",
    [
        (False, signal.SIGTERM),
        (False, _SIGHUP),
        (True, signal.SIGTERM),
        (True, _SIGHUP),
    ],
    ids=["headless-term", "headless-hup", "controller-term", "controller-hup"],
)
@pytest.mark.parametrize(
    "barrier_case",
    [
        ("_dispatch_if_scheduling_open", "before-admission"),
        ("_dispatch_batch", "inside-scheduling-lock"),
    ],
    ids=["before-admission", "inside-scheduling-lock"],
)
def test_signal_during_dispatch_barrier_rejects_new_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: tuple[bool, int],
    barrier_case: tuple[str, str],
) -> None:
    """Delay a real RunLoop method, not the controller or worker implementation."""
    interactive, signum = mode
    method, barrier = barrier_case
    repo, db, pid_file = _project(tmp_path, monkeypatch)
    marker = tmp_path / f"{barrier}-entered"
    release = tmp_path / f"{barrier}-release"
    recorded = tmp_path / f"{barrier}-signal-recorded"
    resumed = tmp_path / f"{barrier}-resumed"
    admitted = tmp_path / f"{barrier}-admitted"
    _ = (tmp_path / "sitecustomize.py").write_text(_DISPATCH_BARRIER, encoding="utf-8")
    monkeypatch.setenv("SHUTDOWN_DISPATCH_METHOD", method)
    monkeypatch.setenv("SHUTDOWN_DISPATCH_MARKER", str(marker))
    monkeypatch.setenv("SHUTDOWN_DISPATCH_RELEASE", str(release))
    monkeypatch.setenv("SHUTDOWN_SIGNAL_RECORDED", str(recorded))
    monkeypatch.setenv("SHUTDOWN_DISPATCH_RESUMED", str(resumed))
    monkeypatch.setenv("SHUTDOWN_DISPATCH_ADMITTED", str(admitted))
    proc, master = _start(repo, pid_file, interactive=interactive, injection=tmp_path)
    try:
        _wait_for(marker)
        started = time.monotonic()
        os.kill(proc.pid, signum)
        _wait_for(recorded, timeout=4)
        release.touch()
        _wait_for(resumed, timeout=4)
        assert not admitted.exists(), "dispatch passed the shutdown admission check"
        assert not pid_file.exists(), "launch passed the shutdown admission barrier"
        with sqlite3.connect(db) as conn:
            assert (
                conn.execute("SELECT 1 FROM nodes WHERE status = 'running'").fetchone() is None
            ), "dispatch claimed a node after shutdown intent"
        assert _wait_exit(proc, master) == 128 + signum, _output(proc, master)
        elapsed = time.monotonic() - started
        assert elapsed < 8.25
    finally:
        _owned_cleanup(proc, pid_file, master)
        release.touch()


@pytest.mark.parametrize(
    "mode",
    [
        (False, signal.SIGTERM),
        (False, _SIGHUP),
        (True, signal.SIGTERM),
        (True, _SIGHUP),
    ],
    ids=["headless-term", "headless-hup", "controller-term", "controller-hup"],
)
@pytest.mark.parametrize(
    "barrier_case",
    [
        ("_dispatch_if_scheduling_open", "before-admission"),
        ("_dispatch_batch", "inside-scheduling-lock"),
    ],
    ids=["before-admission", "inside-scheduling-lock"],
)
def test_signal_exits_while_dispatch_remains_blocked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: tuple[bool, int],
    barrier_case: tuple[str, str],
) -> None:
    interactive, signum = mode
    method, barrier = barrier_case
    repo, db, pid_file = _project(tmp_path, monkeypatch)
    marker = tmp_path / f"{barrier}-entered"
    release = tmp_path / f"{barrier}-release"
    resumed = tmp_path / f"{barrier}-resumed"
    _ = (tmp_path / "sitecustomize.py").write_text(_DISPATCH_BARRIER, encoding="utf-8")
    monkeypatch.setenv("SHUTDOWN_DISPATCH_METHOD", method)
    monkeypatch.setenv("SHUTDOWN_DISPATCH_MARKER", str(marker))
    monkeypatch.setenv("SHUTDOWN_DISPATCH_RELEASE", str(release))
    monkeypatch.setenv("SHUTDOWN_SIGNAL_RECORDED", str(tmp_path / "signal-recorded"))
    monkeypatch.setenv("SHUTDOWN_DISPATCH_RESUMED", str(resumed))
    monkeypatch.setenv("SHUTDOWN_DISPATCH_ADMITTED", str(tmp_path / "admitted"))
    monkeypatch.setenv("SHUTDOWN_BLOCKED_EXIT", "1")
    proc, master = _start(repo, pid_file, interactive=interactive, injection=tmp_path)
    try:
        _wait_for(marker)
        started = time.monotonic()
        os.kill(proc.pid, signum)
        assert _wait_exit(proc, master) == 128 + signum, _output(proc, master)
        assert not release.exists(), "dispatch barrier released before CLI exit"
        assert not resumed.exists(), "admission resumed before CLI exit"
        assert not pid_file.exists(), "launch passed the shutdown admission barrier"
        with sqlite3.connect(db) as conn:
            assert (
                conn.execute("SELECT 1 FROM nodes WHERE status = 'running'").fetchone() is None
            ), "dispatch claimed a node after shutdown intent"
        assert time.monotonic() - started < 8.25
    finally:
        _owned_cleanup(proc, pid_file, master)
        release.touch()
