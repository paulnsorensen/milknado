"""The async worker never settles a run, attempt, or node it no longer owns."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import cast

import pytest

from milknado.domains.common import (
    MilknadoConfig,
    RunResult,
    WorkerIdentity,
    WorkerOwner,
)
from milknado.domains.dispatch import async_run
from milknado.domains.dispatch._async_types import AsyncRunRequest, AsyncWorkerContext
from milknado.domains.dispatch.isolate import MergeBackResult
from milknado.domains.dispatch.ports import ProcessPort
from milknado.domains.graph import FinalizeTaskFailure, MikadoGraph, StartTaskAttempt
from tests.test_mcp_run_enhancements import (
    _branch_git,  # pyright: ignore[reportPrivateUsage]
)

_NOW = "2026-01-01T00:00:00+00:00"
_LOGGER = "milknado.domains.dispatch.async_run"
_KEPT = MergeBackResult(False, "/kept")


class _Sessions:
    _root: Path

    def __init__(self, root: Path) -> None:
        self._root = root

    def open_graph(self, project_root: Path) -> tuple[MikadoGraph, MilknadoConfig]:
        _ = project_root
        db_path = self._root / "graph.db"
        return MikadoGraph(db_path), MilknadoConfig(project_root=self._root, db_path=db_path)


def _world(root: Path) -> tuple[AsyncWorkerContext, MikadoGraph, int]:
    graph = MikadoGraph(root / "graph.db")
    node = graph.add_node("task")
    assert graph.claim_node(node.id, "owner", now=_NOW)
    _ = graph.task_attempts.start(StartTaskAttempt(node.id, "owner", "owner", "owner", 2))
    graph.runs.start("owner", node.id, "run.log", _NOW, 1, attempt_id="owner")
    log_path = root / "run.log"
    log_path.touch()
    request = AsyncRunRequest(
        root, node.id, "brief", None, 1, "owner", "worker", root, max_node_attempts=2
    )
    context = AsyncWorkerContext(
        request=request,
        log_path=log_path,
        argv=("worker",),
        graph_sessions=_Sessions(root),
        git=_branch_git(),
        process=cast(ProcessPort, cast(object, None)),
    )
    return context, graph, node.id


def _sibling_with_open_worker(graph: MikadoGraph, node_id: int) -> None:
    graph.runs.start("linked", node_id, "linked.log", _NOW, 1, attempt_id="owner")
    graph.runs.record_worker(
        WorkerOwner("linked", 2**31 - 1, 123.5, "linked", node_id),
        WorkerIdentity("inv-1", 2**31 - 2, 2**31 - 2, 456.5),
    )


def _play(
    monkeypatch: pytest.MonkeyPatch,
    context: AsyncWorkerContext,
    outcome: tuple[int, bool, bool],
    merge: MergeBackResult | None,
) -> None:
    def worker_process(*_args: object, **_kwargs: object) -> tuple[int, bool, bool]:
        return outcome

    def merge_back(*_args: object, **_kwargs: object) -> MergeBackResult | None:
        return merge

    monkeypatch.setattr(async_run, "run_worker_process", worker_process)
    monkeypatch.setattr(async_run, "merge_back_if_done", merge_back)
    async_run._async_worker(context)  # pyright: ignore[reportPrivateUsage]


def _assert_worker_raised(caplog: pytest.LogCaptureFixture, message: str) -> None:
    texts = [r.getMessage() for r in caplog.records if r.name == _LOGGER]
    assert f"async worker for run owner raised RuntimeError: {message}" in texts


def test_cancelled_run_with_a_settled_attempt_reports_the_lost_attempt_fence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    context, graph, node_id = _world(tmp_path)
    assert graph.task_attempts.confirm_execution("owner", "owner", "owner")
    result = RunResult("failed", 1, False, _NOW)
    _ = graph.task_attempts.fail(FinalizeTaskFailure(node_id, "owner", "owner", "owner", result))
    graph.close()

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        _play(monkeypatch, context, (0, False, True), None)

    _assert_worker_raised(caplog, "task outcome fence lost for attempt owner")


def test_cancelled_run_beside_a_sibling_worker_reports_the_lost_node_fence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    context, graph, node_id = _world(tmp_path)
    _sibling_with_open_worker(graph, node_id)
    graph.close()

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        _play(monkeypatch, context, (0, False, True), None)

    _assert_worker_raised(caplog, "terminal node fence lost for run owner")


def test_failed_run_after_the_node_fence_moved_reports_the_lost_attempt_fence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    context, graph, node_id = _world(tmp_path)
    assert graph.replace_run_id(node_id, "owner", "next")
    graph.close()

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        _play(monkeypatch, context, (1, False, False), None)

    _assert_worker_raised(caplog, "task attempt fence lost for run owner")


def test_preserved_worktree_after_the_node_fence_moved_reports_the_lost_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    context, graph, node_id = _world(tmp_path)
    assert graph.replace_run_id(node_id, "owner", "next")
    graph.close()

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        _play(monkeypatch, context, (0, False, False), _KEPT)

    _assert_worker_raised(caplog, "task outcome fence lost for attempt owner")


def test_preserved_worktree_beside_a_sibling_worker_reports_the_lost_terminal_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    context, graph, node_id = _world(tmp_path)
    _sibling_with_open_worker(graph, node_id)
    graph.close()

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        _play(monkeypatch, context, (0, False, False), _KEPT)

    _assert_worker_raised(caplog, "terminal node fence lost for run owner")


def test_clean_exit_after_the_node_fence_moved_reports_the_lost_done_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    context, graph, node_id = _world(tmp_path)
    assert graph.replace_run_id(node_id, "owner", "next")
    graph.close()

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        _play(monkeypatch, context, (0, False, False), None)

    _assert_worker_raised(caplog, "task outcome fence lost for attempt owner")
