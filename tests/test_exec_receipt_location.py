"""The tmux exec receipt writer and the recovery reader agree on one location."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest

from milknado.adapters.process import ProcessAdapter
from milknado.domains.common import GitPort
from milknado.domains.dispatch import _async_execution
from milknado.domains.dispatch._async_types import AsyncRunRequest, AsyncWorkerContext
from milknado.domains.dispatch._runstate import exec_receipt_path, runs_dir
from milknado.domains.dispatch.ports import GraphSessionPort, RunWindow, TmuxPort
from milknado.domains.dispatch.reconcile import fail_stale_running_runs
from milknado.domains.graph import MikadoGraph, StartTaskAttempt

pytestmark = pytest.mark.real_process

_NOW = "2026-01-01T00:00:00+00:00"


def test_recovery_finds_receipt_for_custom_db_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    process = subprocess.Popen([sys.executable, "-c", "pass"])
    assert process.wait(timeout=3) == 0
    project = tmp_path / "project"
    project.mkdir()
    state = tmp_path / "state"
    state.mkdir()
    graph = MikadoGraph(state / "graph.db")
    node = graph.add_node("task")
    assert graph.claim_node(node.id, "owner", now=_NOW, pid=process.pid)
    _ = graph.task_attempts.start(StartTaskAttempt(node.id, "attempt", "owner", "owner", 2))
    graph.runs.start("owner", node.id, "run.log", _NOW, 1, attempt_id="attempt")
    graph.runs.set_pid("owner", process.pid)
    windows: list[RunWindow] = []

    def capture(_tmux: object, window: RunWindow, *_args: object, **_kwargs: object):
        windows.append(window)
        return 0, False, False

    monkeypatch.setattr(_async_execution, "_execute_in_window", capture)
    request = AsyncRunRequest(project, node.id, "brief", None, 1, "owner", "claude", project)
    context = AsyncWorkerContext(
        request,
        tmp_path / "w.log",
        ("claude",),
        cast(GraphSessionPort, object()),
        cast(GitPort, object()),
        ProcessAdapter(),
        tmux=cast(TmuxPort, object()),
    )
    _ = _async_execution.run_worker_process(context, graph, runs_dir(project))
    receipt = windows[0].exec_receipt_path
    assert receipt is not None
    nonce = windows[0].exec_nonce
    body = {"nonce": nonce, "state": "FINISHED", "pid": process.pid, "returncode": 1}
    _ = receipt.write_text(json.dumps(body))

    assert len(fail_stale_running_runs(graph, node.id, ProcessAdapter())) == 1
    attempt = graph.task_attempts.get("attempt")
    assert attempt is not None and attempt.start_state == "executed"
    graph.close()


def test_exec_receipt_path_is_pure(tmp_path: Path) -> None:
    receipt = exec_receipt_path(tmp_path / "graph.db", "run", "nonce")

    assert receipt == tmp_path / "runs" / "run.nonce.exec.json"
    assert not (tmp_path / "runs").exists()
