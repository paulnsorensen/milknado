"""Each gate descriptor is handed off exactly once, and the gate runs without cwd imports."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast, final
from unittest import mock

import pytest
from typing_extensions import override

from milknado.domains.common import WorkerIdentity
from milknado.loop._process_gate import SpawnOptions, WorkerProcess, spawn_gated

pytestmark = pytest.mark.skipif(os.name == "nt", reason="POSIX exec gate is required")


@dataclass(frozen=True)
class _Pause:
    on_read: threading.Event = field(default_factory=threading.Event)
    resume: threading.Event = field(default_factory=threading.Event)


@final
class _SlowRead(WorkerProcess):  # pyright: ignore[reportGeneralTypeIssues]
    """Pauses the releasing thread right after it first reads the gate descriptor."""

    def __init__(
        self,
        process: subprocess.Popen[bytes],
        identity: WorkerIdentity,
        gate_fd: int,
        pause: _Pause,
    ) -> None:
        self._pause: _Pause = pause
        self._paused: bool = False
        super().__init__(process, identity, gate_fd)

    @property
    @override
    def _gate_fd(self) -> int | None:
        value: int | None = self.__dict__.get("gate")
        if threading.current_thread().name == "releaser" and not self._paused and value:
            self._paused = True
            self._pause.on_read.set()
            _ = self._pause.resume.wait(timeout=0.5)
        return value

    @_gate_fd.setter
    def _gate_fd(self, value: int | None) -> None:
        self.__dict__["gate"] = value


def test_concurrent_close_and_release_close_each_fd_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate_read, gate_write = os.pipe()
    pause = _Pause()
    closed: list[int] = []
    errors: list[BaseException] = []
    real_close = os.close

    def recording_close(fd: int) -> None:
        closed.append(fd)
        real_close(fd)

    process = subprocess.Popen([sys.executable, "-c", "pass"])
    identity = WorkerIdentity("inv", process.pid, process.pid, 0.0)
    worker = _SlowRead(process, identity, gate_write, pause)
    monkeypatch.setattr(os, "close", recording_close)

    def release() -> None:
        try:
            worker.release()
        except Exception as exc:
            errors.append(exc)

    releaser = threading.Thread(target=release, name="releaser")
    closer = threading.Thread(target=worker.close_gate)
    releaser.start()
    assert pause.on_read.wait(timeout=2)
    closer.start()
    closer.join(timeout=0.2)
    pause.resume.set()
    releaser.join(timeout=5)
    closer.join(timeout=5)
    real_close(gate_read)
    _ = process.wait(timeout=5)

    assert errors == []
    assert closed.count(gate_write) == 1, f"gate fd closed {closed.count(gate_write)} times"


def test_gate_child_runs_isolated_from_cwd_imports(tmp_path: Path) -> None:
    with mock.patch.object(subprocess, "Popen", wraps=subprocess.Popen) as popen:
        worker = spawn_gated(
            SpawnOptions(
                (sys.executable, "-c", "pass"),
                tmp_path,
                dict(os.environ),
                True,
                subprocess.DEVNULL,
                subprocess.DEVNULL,
                subprocess.DEVNULL,
            )
        )
    try:
        worker.release()
        _ = worker.process.wait(timeout=10)
    finally:
        worker.close_gate()
    command = tuple(cast(Sequence[str], popen.call_args_list[0].args[0]))
    assert command[:3] == (sys.executable, "-P", "-m")
