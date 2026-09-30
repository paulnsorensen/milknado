"""Failure paths for the lifeline helper handshake."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

import milknado.loop._process_helper as process_helper
from milknado.loop._process_contract import ProtectionContext
from milknado.loop._process_gate import WorkerProcess
from milknado.loop._process_helper import (
    HelperStart,
    _await_ready,  # pyright: ignore[reportPrivateUsage]
    stop_failed_helper,
)


@pytest.mark.skipif(os.name == "nt", reason="POSIX pipe semantics are required")
@pytest.mark.parametrize(
    "frame", [b"\xff\n", b"READY wrong\n", b"READY expected\nextra", b"x" * 256]
)
def test_ready_rejects_invalid_frame(frame: bytes) -> None:
    helper = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import os,sys; os.write(1, bytes.fromhex(sys.argv[1]))",
            frame.hex(),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert not _await_ready(helper, "READY expected", time.monotonic() + 2)
        assert helper.wait(timeout=2) == 0
    finally:
        if helper.poll() is None:
            helper.kill()
            _ = helper.wait(timeout=2)
        assert helper.stdout is not None
        helper.stdout.close()
        assert helper.stderr is not None
        helper.stderr.close()


@pytest.mark.skipif(os.name == "nt", reason="POSIX pipe semantics are required")
def test_ready_requires_stdout_pipe() -> None:
    helper = subprocess.Popen(
        [sys.executable, "-c", "pass"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert not _await_ready(helper, "READY expected", time.monotonic() + 2)
        assert helper.wait(timeout=2) == 0
    finally:
        if helper.poll() is None:
            helper.kill()
            _ = helper.wait(timeout=2)
        assert helper.stderr is not None
        helper.stderr.close()


@pytest.mark.skipif(os.name == "nt", reason="POSIX signals are required")
def test_failed_helper_escalates_when_sigterm_ignored(tmp_path: Path) -> None:
    ready = tmp_path / "ready"
    helper = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import signal,sys,time; from pathlib import Path; "
            + "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            + "Path(sys.argv[1]).touch(); time.sleep(30)",
            str(ready),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 3
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.exists()
        stop_failed_helper(helper, deadline)
        assert helper.returncode == -signal.SIGKILL
        assert helper.stdout is not None and helper.stdout.closed
        assert helper.stderr is not None and helper.stderr.closed
    finally:
        if helper.poll() is None:
            helper.kill()
            _ = helper.wait(timeout=2)


@pytest.mark.skipif(os.name == "nt", reason="POSIX inherited pipe is required")
def test_helper_spawn_failure_closes_both_pipe_fds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_pipe = os.pipe
    owned: list[int] = []

    def tracked_pipe() -> tuple[int, int]:
        pair = real_pipe()
        owned.extend(pair)
        return pair

    def fail_popen(*_args: object, **_kwargs: object) -> None:
        raise OSError("helper spawn denied")

    monkeypatch.setattr(os, "pipe", tracked_pipe)
    monkeypatch.setattr(subprocess, "Popen", fail_popen)
    with pytest.raises(OSError, match="helper spawn denied"):
        _ = process_helper.start_helper(
            cast(
                WorkerProcess,
                cast(object, SimpleNamespace(identity=SimpleNamespace(invocation_id="run"))),
            ),
            cast(ProtectionContext, cast(object, SimpleNamespace(db_path=Path("unused.db")))),
            HelperStart(0, 0, time.monotonic() + 2),
        )
    assert len(owned) == 2
    for fd in owned:
        with pytest.raises(OSError):
            _ = os.fstat(fd)
