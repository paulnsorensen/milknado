"""Synchronous dispatch refuses to settle attempts it cannot prove it still owns."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from milknado.domains.common import NodeStatus, WorkerIdentity, WorkerOwner, WorktreeMode
from milknado.domains.dispatch import lifecycle
from milknado.domains.dispatch.lifecycle import SyncDispatchRequest
from milknado.domains.dispatch.ports import ProcessPort, WorkerOutcomePort
from milknado.domains.graph import MikadoGraph, StartTaskAttempt
from tests.test_mcp_run_enhancements import (
    _branch_git,  # pyright: ignore[reportPrivateUsage]
)

_NOW = "2026-01-01T00:00:00+00:00"
_WORKER = cast(WorkerOutcomePort, cast(object, SimpleNamespace(exit_code=1, timed_out=False)))


def _attempt(tmp_path: Path, limit: int = 2) -> tuple[MikadoGraph, int]:
    graph = MikadoGraph(tmp_path / "graph.db")
    node = graph.add_node("task")
    assert graph.claim_node(node.id, "owner", now=_NOW)
    _ = graph.task_attempts.start(StartTaskAttempt(node.id, "attempt", "owner", "owner", limit))
    graph.runs.start("owner", node.id, "run.log", _NOW, None, attempt_id="attempt")
    return graph, node.id


def _open_worker(graph: MikadoGraph, run_id: str, node_id: int) -> None:
    graph.runs.record_worker(
        WorkerOwner(run_id, 2**31 - 1, 123.5, run_id, node_id),
        WorkerIdentity("inv-1", 2**31 - 2, 2**31 - 2, 456.5),
    )


def test_finish_refuses_while_the_run_still_has_a_worker(tmp_path: Path) -> None:
    graph, node_id = _attempt(tmp_path)
    _open_worker(graph, "owner", node_id)

    with pytest.raises(RuntimeError, match="worker cleanup unresolved for run owner"):
        lifecycle._finish_dispatch(  # pyright: ignore[reportPrivateUsage]
            graph, node_id, "owner", _WORKER, "failed", None
        )

    run = graph.runs.get("owner")
    assert run is not None and run["status"] == "running"
    graph.close()


def test_counted_failure_rejects_a_run_whose_attempt_row_vanished(tmp_path: Path) -> None:
    graph, node_id = _attempt(tmp_path)
    with closing(sqlite3.connect(graph.db_path)) as raw:
        _ = raw.execute("DELETE FROM task_attempts WHERE attempt_id = 'attempt'")
        raw.commit()

    with pytest.raises(ValueError, match="task attempt missing for run owner"):
        _ = lifecycle._fail_counted_attempt(  # pyright: ignore[reportPrivateUsage]
            graph, node_id, "owner", _WORKER, None
        )
    graph.close()


def test_counted_failure_rejects_a_lost_owner(tmp_path: Path) -> None:
    graph, node_id = _attempt(tmp_path)
    assert graph.replace_run_id(node_id, "owner", "next")

    with pytest.raises(RuntimeError, match="task attempt fence lost for run owner"):
        _ = lifecycle._fail_counted_attempt(  # pyright: ignore[reportPrivateUsage]
            graph, node_id, "owner", _WORKER, None
        )

    assert graph.task_attempts.failed_count(node_id) == 0
    graph.close()


def test_nonfailure_outcome_rejects_a_lost_owner(tmp_path: Path) -> None:
    graph, node_id = _attempt(tmp_path)
    assert graph.replace_run_id(node_id, "owner", "next")

    with pytest.raises(RuntimeError, match="task attempt outcome fence lost for run owner"):
        lifecycle._record_attempt_nonfailure(  # pyright: ignore[reportPrivateUsage]
            graph, "owner", "done"
        )

    attempt = graph.task_attempts.get("attempt")
    assert attempt is not None and attempt.outcome == "active"
    graph.close()


def test_sync_dispatch_rejects_worker_left_behind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    node = graph.add_node("task")
    monkeypatch.setattr(lifecycle, "render_brief", lambda *_args, **_kwargs: "")  # pyright: ignore[reportUnknownArgumentType, reportUnknownLambdaType]

    def leave_worker(*_args: object, **kwargs: object) -> SimpleNamespace:
        _open_worker(graph, cast(str, kwargs["run_id"]), node.id)
        return SimpleNamespace(exit_code=0, timed_out=False)

    monkeypatch.setattr(lifecycle, "run_headless", leave_worker)
    request = SyncDispatchRequest(
        node_id=node.id,
        project_root=tmp_path,
        worker_cmd=None,
        timeout_seconds=1,
        default_cmd="claude",
        process=cast(ProcessPort, object()),
        worktree_mode=WorktreeMode.THIS_BRANCH,
    )

    with pytest.raises(RuntimeError, match="worker cleanup unresolved"):
        _ = lifecycle.dispatch_node_sync(graph, _branch_git(), request)

    current = graph.get_node(node.id)
    assert current is not None and current.status is NodeStatus.RUNNING
    graph.close()
