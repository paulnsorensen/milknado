"""Tests for run_loop/_completion.py: handle_completion paths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

from milknado.domains.common.types import MikadoNode, NodeStatus
from milknado.domains.execution.executor import RebaseConflict
from milknado.domains.execution.run_loop._completion import handle_completion
from milknado.domains.execution.run_loop.input import InputState


@dataclass
class _FakeCompleteResult:
    rebase_conflict: RebaseConflict | None = None
    redispatch: Any | None = None
    blocked: bool = False
    review_notification_failed: bool = False


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
        self._complete_result = _FakeCompleteResult()

    def complete(self, node_id: int, feature_branch: str) -> _FakeCompleteResult:
        return self._complete_result

    def fail(self, node_id: int, detail: str | None = None) -> None:
        self.failed_ids.append(node_id)
        self.fail_details.append(detail)


def _make_node(node_id: int, desc: str = "task") -> MikadoNode:
    return MikadoNode(id=node_id, description=desc, status=NodeStatus.RUNNING)


def _make_loop(node_id: int, run_id: str, executor: _FakeExecutor) -> Any:
    """Build a minimal duck-typed RunLoop-like object for handle_completion."""
    graph = _FakeGraph()
    graph.add(_make_node(node_id))

    loop = MagicMock()
    loop._active = {run_id: node_id}
    loop._progress_by_run = {}
    loop._input = InputState()
    loop._graph = graph
    loop._executor = executor
    loop._completion_durations = []
    loop._dispatched_at = {}
    loop._logs = []
    loop._attempts = {}
    loop._strict = False
    loop._failure_triggered = False
    return loop


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

        handle_completion(loop, "run-1", "completed", "main", live)

        assert "run-1" not in loop._active

    def test_appends_duration_on_success(self) -> None:
        exec_ = _FakeExecutor()
        loop = _make_loop(1, "run-1", exec_)
        live = MagicMock()

        handle_completion(loop, "run-1", "completed", "main", live)

        assert len(loop._completion_durations) == 1

    def test_clears_overlay_when_run_is_active(self) -> None:
        exec_ = _FakeExecutor()
        loop = _make_loop(1, "run-1", exec_)
        loop._input.overlay_state = "run-1"
        live = MagicMock()

        handle_completion(loop, "run-1", "completed", "main", live)

        assert loop._input.overlay_state is None


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

        handle_completion(loop, "run-1", "failed", "main", live)

        assert 1 in exec_.failed_ids

    def test_strict_sets_failure_triggered(self) -> None:
        exec_ = _FakeExecutor()
        loop = _make_loop(1, "run-1", exec_)
        loop._strict = True
        live = MagicMock()

        handle_completion(loop, "run-1", "failed", "main", live)

        assert loop._failure_triggered is True

    def test_non_strict_does_not_set_failure_triggered(self) -> None:
        exec_ = _FakeExecutor()
        loop = _make_loop(1, "run-1", exec_)
        loop._strict = False
        live = MagicMock()

        handle_completion(loop, "run-1", "failed", "main", live)

        assert loop._failure_triggered is False

    def test_failure_log_includes_last_agent_output(self, caplog) -> None:
        exec_ = _FakeExecutor()
        loop = _make_loop(1, "run-1", exec_)
        loop._ralph.get_run_failure_detail.return_value = "Error: unknown flag: --mcp-config"
        live = MagicMock()
        caplog.set_level("WARNING", logger="milknado")

        handle_completion(loop, "run-1", False, "main", live)

        assert "unknown flag: --mcp-config" in caplog.text
        # The real failure detail is threaded into fail() so the runs row
        # keeps it, not a static "node failed".
        assert exec_.fail_details == ["Error: unknown flag: --mcp-config"]

    def test_failure_log_bare_when_no_detail(self, caplog) -> None:
        exec_ = _FakeExecutor()
        loop = _make_loop(1, "run-1", exec_)
        loop._ralph.get_run_failure_detail.return_value = None
        live = MagicMock()
        caplog.set_level("WARNING", logger="milknado")

        handle_completion(loop, "run-1", False, "main", live)

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
        exec_._complete_result = _FakeCompleteResult(rebase_conflict=conflict)
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
        exec_._complete_result = _FakeCompleteResult(rebase_conflict=conflict)
        loop = _make_loop(1, "run-1", exec_)
        loop._strict = True
        live = MagicMock()

        handle_completion(loop, "run-1", "completed", "main", live)

        assert loop._failure_triggered is True

    def test_no_conflict_does_not_append_to_conflicts(self) -> None:
        exec_ = _FakeExecutor()
        loop = _make_loop(1, "run-1", exec_)
        live = MagicMock()

        _, _, cs = handle_completion(loop, "run-1", "completed", "main", live)

        assert cs == []
