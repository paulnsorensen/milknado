from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path
from threading import Thread
from typing import IO, cast

import pytest

from milknado.adapters import GitAdapter
from milknado.adapters import process as process_module
from milknado.adapters.process import ProcessAdapter
from milknado.domains.dispatch.cancel import cancel_run
from milknado.domains.graph import MikadoGraph


def test_term_resistant_group_has_kill_confirmation_within_cancel_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [0.0]
    killed_at: list[float] = []

    def killpg(_pgid: int, signum: int) -> None:
        if signum == signal.SIGKILL:
            killed_at.append(clock[0])

    def alive(_pid: int) -> bool:
        return not killed_at or clock[0] < killed_at[0] + 0.2

    def sleep(seconds: float) -> None:
        clock[0] += seconds

    def getpgid(pid: int) -> int:
        return pid

    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(time, "sleep", sleep)
    monkeypatch.setattr(process_module, "pid_alive", alive)
    monkeypatch.setattr(os, "getpgid", getpgid)
    monkeypatch.setattr(os, "killpg", killpg)

    terminated = ProcessAdapter(poll_interval=0.1).terminate_group(123, 2.0)
    assert terminated
    assert len(killed_at) == 1
    assert killed_at[0] < 1.8
    assert clock[0] <= 2.0


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups required")
def test_cancel_kills_term_resistant_supervisor_before_finalizing(tmp_path: Path) -> None:
    script = (
        "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        + "print('ready', flush=True); time.sleep(60)"
    )
    proc: subprocess.Popen[str] = subprocess.Popen(
        [sys.executable, "-c", script],
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    graph = MikadoGraph(tmp_path / "graph.db")
    try:
        assert proc.stdout is not None
        ready = cast(IO[str], proc.stdout).readline().strip()
        assert ready == "ready"
        node = graph.add_node("cancel resistant supervisor")
        claimed = graph.claim_node(node.id, "run-1", now="2026-01-01T00:00:00+00:00", pid=proc.pid)
        assert claimed
        graph.runs.start("run-1", node.id, "worker.log", "2026-01-01T00:00:00+00:00", None)
        graph.runs.set_pid("run-1", proc.pid)
        reaper = Thread(target=proc.wait, daemon=True)
        reaper.start()

        started = time.monotonic()
        result = cancel_run(graph, GitAdapter(tmp_path), ProcessAdapter(), tmp_path, "run-1")
        elapsed = time.monotonic() - started

        reaper.join(timeout=1)
        assert result["status"] == "failed"
        assert proc.returncode == -signal.SIGKILL
        assert elapsed < 8.0
    finally:
        with suppress(ProcessLookupError, PermissionError):
            os.killpg(proc.pid, signal.SIGKILL)
        _ = proc.wait(timeout=2)
        graph.close()
        if proc.stdout is not None:
            proc.stdout.close()
