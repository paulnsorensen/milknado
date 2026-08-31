"""Tests for run_loop/_completion.py: handle_completion paths."""

from __future__ import annotations

from collections import deque
from typing import cast
from unittest.mock import MagicMock

import pytest

from milknado.domains.common.protocols import LoopPort, ProgressEvent
from milknado.domains.common.types import MikadoNode, NodeStatus
from milknado.domains.execution.executor import (
    CompletionResult,
    Executor,
    RebaseConflict,
)
from milknado.domains.execution.run_loop._completion import handle_completion
from milknado.domains.execution.run_loop.input import InputState
from milknado.domains.execution.run_loop.state import TerminalRunState
from milknado.domains.graph import MikadoGraph


class _FakeGraph:
    def __init__(self) -> None:
        self._nodes: dict[int, MikadoNode] = {}

    def add(self, node: MikadoNode) -> None:
        self._nodes[node.id] = node

    def get_node(self, node_id: int) -> MikadoNode | None:
        return self._nodes.get(node_id)


class _FakeExecutor:
    def __init__(self) -> None:
        self.failed_ids: list[int] = []
        self.fail_details: list[str | None] = []
        self.complete_result: CompletionResult = CompletionResult(
            node_id=0, rebased=False, newly_ready=[]
        )

    def complete(self, _node_id: int, _feature_branch: str) -> CompletionResult:
        return self.complete_result

    def fail(self, node_id: int, detail: str | None = None) -> None:
        self.failed_ids.append(node_id)
        self.fail_details.append(detail)

    def cancel(self, _node_id: int) -> None:
        return None


def _make_node(node_id: int, desc: str = "task") -> MikadoNode:
    return MikadoNode(id=node_id, description=desc, status=NodeStatus.RUNNING)


class _FakeRalph:
    def __init__(self) -> None:
        self.failure_detail: str | None = None

    def get_run_failure_detail(self, _run_id: str) -> str | None:
        return self.failure_detail

    def get_run_output_tail(self, _run_id: str, _max_lines: int) -> list[str]:
        return []

    def get_run_guidance(self, _run_id: str) -> tuple[str, ...]:
        return ()


class _FakeLoop:
    _active: dict[str, int]
    _progress_by_run: dict[str, ProgressEvent]
    _input: InputState
    _graph: MikadoGraph
    _dispatched_at: dict[str, float]
    _logs: deque[str]
    _executor: Executor
    _completion_durations: deque[float]
    _ralph: LoopPort
    _attempts: dict[int, int]
    _strict: bool
    _failure_triggered: bool
    _stopped_nodes: set[int]
    _stopped: int
    _terminal_runs: deque[TerminalRunState]

    def __init__(self, node_id: int, run_id: str, executor: _FakeExecutor) -> None:
        graph = _FakeGraph()
        graph.add(_make_node(node_id))
        self._active = {run_id: node_id}
        self._progress_by_run = {}
        self._input = InputState()
        self._graph = cast(MikadoGraph, cast(object, graph))
        self._executor = cast(Executor, cast(object, executor))
        self._completion_durations = deque()
        self._dispatched_at = {}
        self._logs = deque()
        self._ralph = cast(LoopPort, cast(object, _FakeRalph()))
        self._attempts = {}
        self._strict = False
        self._failure_triggered = False
        self._stopped_nodes = set()
        self._stopped = 0
        self._terminal_runs = deque()


def _make_loop(node_id: int, run_id: str, executor: _FakeExecutor) -> _FakeLoop:
    return _FakeLoop(node_id, run_id, executor)


class TestHandleCompletionSuccess:
    def test_returns_one_completed_zero_failed(self) -> None:
        exec_ = _FakeExecutor()
        loop = _make_loop(1, "run-1", exec_)
        live = MagicMock()

        c, f, cs = handle_completion(loop, "run-1", "completed", "main", live)

        assert c == 1
        assert f == 0
        assert cs == []

    def test_removes_run_from_active(self) -> None:
        exec_ = _FakeExecutor()
        loop = _make_loop(1, "run-1", exec_)
        live = MagicMock()

        _ = handle_completion(loop, "run-1", "completed", "main", live)

        assert "run-1" not in loop._active  # pyright: ignore[reportPrivateUsage]

    def test_appends_duration_on_success(self) -> None:
        exec_ = _FakeExecutor()
        loop = _make_loop(1, "run-1", exec_)
        live = MagicMock()

        _ = handle_completion(loop, "run-1", "completed", "main", live)

        assert len(loop._completion_durations) == 1  # pyright: ignore[reportPrivateUsage]

    def test_clears_overlay_when_run_is_active(self) -> None:
        exec_ = _FakeExecutor()
        loop = _make_loop(1, "run-1", exec_)
        loop._input.overlay_state = "run-1"  # pyright: ignore[reportPrivateUsage]
        live = MagicMock()

        _ = handle_completion(loop, "run-1", "completed", "main", live)

        assert loop._input.overlay_state is None  # pyright: ignore[reportPrivateUsage]


class TestHandleCompletionFailure:
    def test_returns_zero_completed_one_failed(self) -> None:
        exec_ = _FakeExecutor()
        loop = _make_loop(1, "run-1", exec_)
        live = MagicMock()

        c, f, cs = handle_completion(loop, "run-1", "failed", "main", live)

        assert c == 0
        assert f == 1
        assert cs == []

    def test_calls_executor_fail(self) -> None:
        exec_ = _FakeExecutor()
        loop = _make_loop(1, "run-1", exec_)
        live = MagicMock()

        _ = handle_completion(loop, "run-1", "failed", "main", live)

        assert 1 in exec_.failed_ids

    def test_strict_sets_failure_triggered(self) -> None:
        exec_ = _FakeExecutor()
        loop = _make_loop(1, "run-1", exec_)
        loop._strict = True  # pyright: ignore[reportPrivateUsage]
        live = MagicMock()

        _ = handle_completion(loop, "run-1", "failed", "main", live)

        assert loop._failure_triggered is True  # pyright: ignore[reportPrivateUsage]

    def test_non_strict_does_not_set_failure_triggered(self) -> None:
        exec_ = _FakeExecutor()
        loop = _make_loop(1, "run-1", exec_)
        loop._strict = False  # pyright: ignore[reportPrivateUsage]
        live = MagicMock()

        _ = handle_completion(loop, "run-1", "failed", "main", live)

        assert loop._failure_triggered is False  # pyright: ignore[reportPrivateUsage]

    def test_failure_log_includes_last_agent_output(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        exec_ = _FakeExecutor()
        loop = _make_loop(1, "run-1", exec_)
        cast(
            _FakeRalph,
            cast(object, loop._ralph),  # pyright: ignore[reportPrivateUsage]
        ).failure_detail = "Error: unknown flag: --mcp-config"
        live = MagicMock()
        caplog.set_level("WARNING", logger="milknado")

        _ = handle_completion(loop, "run-1", "failed", "main", live)

        assert "unknown flag: --mcp-config" in caplog.text
        # The real failure detail is threaded into fail() so the runs row
        # keeps it, not a static "node failed".
        assert exec_.fail_details == ["Error: unknown flag: --mcp-config"]

    def test_failure_log_bare_when_no_detail(self, caplog: pytest.LogCaptureFixture) -> None:
        exec_ = _FakeExecutor()
        loop = _make_loop(1, "run-1", exec_)
        cast(_FakeRalph, cast(object, loop._ralph)).failure_detail = None  # pyright: ignore[reportPrivateUsage]
        live = MagicMock()
        caplog.set_level("WARNING", logger="milknado")

        _ = handle_completion(loop, "run-1", "failed", "main", live)

        assert "node_failed node_id=1" in caplog.text
        assert "detail=" not in caplog.text
        assert exec_.fail_details == [None]


class TestHandleCompletionRebaseConflict:
    def test_conflict_counted_as_failed(self) -> None:
        exec_ = _FakeExecutor()
        conflict = RebaseConflict(
            node_id=1,
            description="task",
            conflicting_files=("src/a.py",),
            detail="CONFLICT in src/a.py",
        )
        exec_.complete_result = CompletionResult(
            node_id=1, rebased=False, newly_ready=[], rebase_conflict=conflict
        )
        loop = _make_loop(1, "run-1", exec_)
        live = MagicMock()

        c, f, cs = handle_completion(loop, "run-1", "completed", "main", live)

        assert c == 0
        assert f == 1
        assert len(cs) == 1
        assert cs[0].node_id == 1

    def test_conflict_strict_sets_failure_triggered(self) -> None:
        exec_ = _FakeExecutor()
        conflict = RebaseConflict(
            node_id=1, description="task", conflicting_files=("src/x.py",), detail=""
        )
        exec_.complete_result = CompletionResult(
            node_id=1, rebased=False, newly_ready=[], rebase_conflict=conflict
        )
        loop = _make_loop(1, "run-1", exec_)
        loop._strict = True  # pyright: ignore[reportPrivateUsage]
        live = MagicMock()

        _ = handle_completion(loop, "run-1", "completed", "main", live)

        assert loop._failure_triggered is True  # pyright: ignore[reportPrivateUsage]

    def test_no_conflict_does_not_append_to_conflicts(self) -> None:
        exec_ = _FakeExecutor()
        loop = _make_loop(1, "run-1", exec_)
        live = MagicMock()

        _, _, cs = handle_completion(loop, "run-1", "completed", "main", live)

        assert cs == []
