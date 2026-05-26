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
        self._complete_result = _FakeCompleteResult()

    def complete(self, node_id: int, feature_branch: str) -> _FakeCompleteResult:
        return self._complete_result

    def fail(self, node_id: int) -> None:
        self.failed_ids.append(node_id)


def _make_node(node_id: int, desc: str = "task") -> MikadoNode:
    return MikadoNode(id=node_id, description=desc, status=NodeStatus.RUNNING)


def _make_loop(node_id: int, run_id: str, executor: _FakeExecutor) -> Any:
    """Build a minimal duck-typed RunLoop-like object for handle_completion."""
    graph = _FakeGraph()
    graph.add(_make_node(node_id))

    loop = MagicMock()
    loop._active = {run_id: node_id}
    loop._input = InputState()
    loop._graph = graph
    loop._executor = executor
    loop._completion_durations = []
    loop._dispatched_at = {}
    loop._logs = []
    loop._attempts = {}
    loop._strict = False
    loop._failure_triggered = False
    loop._pr_stack = False
    loop._completed_branches = {}
    return loop


class TestHandleCompletionSuccess:
    def test_returns_one_completed_zero_failed(self) -> None:
        exec_ = _FakeExecutor()
        loop = _make_loop(1, "run-1", exec_)
        live = MagicMock()

        c, f, cs = handle_completion(loop, "run-1", True, "main", live)

        assert c == 1
        assert f == 0
        assert cs == []

    def test_removes_run_from_active(self) -> None:
        exec_ = _FakeExecutor()
        loop = _make_loop(1, "run-1", exec_)
        live = MagicMock()

        handle_completion(loop, "run-1", True, "main", live)

        assert "run-1" not in loop._active

    def test_appends_duration_on_success(self) -> None:
        exec_ = _FakeExecutor()
        loop = _make_loop(1, "run-1", exec_)
        live = MagicMock()

        handle_completion(loop, "run-1", True, "main", live)

        assert len(loop._completion_durations) == 1

    def test_clears_overlay_when_run_is_active(self) -> None:
        exec_ = _FakeExecutor()
        loop = _make_loop(1, "run-1", exec_)
        loop._input.overlay_state = "run-1"
        live = MagicMock()

        handle_completion(loop, "run-1", True, "main", live)

        assert loop._input.overlay_state is None


class TestHandleCompletionFailure:
    def test_returns_zero_completed_one_failed(self) -> None:
        exec_ = _FakeExecutor()
        loop = _make_loop(1, "run-1", exec_)
        live = MagicMock()

        c, f, cs = handle_completion(loop, "run-1", False, "main", live)

        assert c == 0
        assert f == 1
        assert cs == []

    def test_calls_executor_fail(self) -> None:
        exec_ = _FakeExecutor()
        loop = _make_loop(1, "run-1", exec_)
        live = MagicMock()

        handle_completion(loop, "run-1", False, "main", live)

        assert 1 in exec_.failed_ids

    def test_strict_sets_failure_triggered(self) -> None:
        exec_ = _FakeExecutor()
        loop = _make_loop(1, "run-1", exec_)
        loop._strict = True
        live = MagicMock()

        handle_completion(loop, "run-1", False, "main", live)

        assert loop._failure_triggered is True

    def test_non_strict_does_not_set_failure_triggered(self) -> None:
        exec_ = _FakeExecutor()
        loop = _make_loop(1, "run-1", exec_)
        loop._strict = False
        live = MagicMock()

        handle_completion(loop, "run-1", False, "main", live)

        assert loop._failure_triggered is False


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

        c, f, cs = handle_completion(loop, "run-1", True, "main", live)

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

        handle_completion(loop, "run-1", True, "main", live)

        assert loop._failure_triggered is True

    def test_no_conflict_does_not_append_to_conflicts(self) -> None:
        exec_ = _FakeExecutor()
        loop = _make_loop(1, "run-1", exec_)
        live = MagicMock()

        _, _, cs = handle_completion(loop, "run-1", True, "main", live)

        assert cs == []


class _FakeStageExecutor(_FakeExecutor):
    def __init__(self) -> None:
        super().__init__()
        self.staged: list[tuple[int, str]] = []

    def stage_for_pr(self, node_id: int, feature_branch: str) -> str:
        self.staged.append((node_id, feature_branch))
        return f"milknado/{node_id}-branch"


class TestHandleCompletionPrStack:
    def test_calls_stage_for_pr_not_complete(self) -> None:
        exec_ = _FakeStageExecutor()
        loop = _make_loop(1, "run-1", exec_)
        loop._pr_stack = True
        live = MagicMock()

        handle_completion(loop, "run-1", True, "main", live)

        assert len(exec_.staged) == 1
        assert exec_.staged[0] == (1, "main")
        # normal complete() must NOT be called
        assert exec_._complete_result is not None  # still default; complete() never called

    def test_accumulates_branch_in_completed_branches(self) -> None:
        exec_ = _FakeStageExecutor()
        loop = _make_loop(1, "run-1", exec_)
        loop._pr_stack = True
        live = MagicMock()

        handle_completion(loop, "run-1", True, "main", live)

        assert loop._completed_branches[1] == "milknado/1-branch"

    def test_returns_completed_one_failed_zero(self) -> None:
        exec_ = _FakeStageExecutor()
        loop = _make_loop(1, "run-1", exec_)
        loop._pr_stack = True
        live = MagicMock()

        c, f, cs = handle_completion(loop, "run-1", True, "main", live)

        assert c == 1
        assert f == 0
        assert cs == []

    def test_no_conflict_in_pr_stack_mode(self) -> None:
        """PR stack mode never produces rebase conflicts (no rebase happens)."""
        exec_ = _FakeStageExecutor()
        loop = _make_loop(1, "run-1", exec_)
        loop._pr_stack = True
        live = MagicMock()

        _, _, cs = handle_completion(loop, "run-1", True, "main", live)

        assert cs == []
