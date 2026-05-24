from __future__ import annotations

from pathlib import Path

from milknado.domains.common.errors import CompletionTimeout
from milknado.domains.execution import run_node_to_completion
from milknado.domains.execution.executor import (
    CompletionResult,
    DispatchResult,
    RebaseConflict,
)


class _FakeExecutor:
    def __init__(self, completion: CompletionResult | None = None) -> None:
        self._completion = completion
        self.dispatched: list[int] = []
        self.completed: list[int] = []
        self.failed: list[int] = []

    def dispatch(self, node_id: int, config) -> DispatchResult:  # noqa: ANN001
        self.dispatched.append(node_id)
        return DispatchResult(node_id=node_id, worktree=Path("/tmp/wt"), run_id=f"run-{node_id}")

    def complete(self, node_id: int, feature_branch: str) -> CompletionResult:
        self.completed.append(node_id)
        assert self._completion is not None
        return self._completion

    def fail(self, node_id: int) -> None:
        self.failed.append(node_id)


class _FakeRalph:
    def __init__(self, *, completed: bool = True, timeout: bool = False) -> None:
        self._completed = completed
        self._timeout = timeout

    def wait_for_next_completion(self, active_run_ids, timeout=None):  # noqa: ANN001
        if self._timeout:
            raise CompletionTimeout(active_run_ids=active_run_ids, waited_seconds=timeout or 0.0)
        return next(iter(active_run_ids)), self._completed


def _ok_completion(node_id: int) -> CompletionResult:
    return CompletionResult(node_id=node_id, rebased=True, newly_ready=[], rebase_conflict=None)


def test_success_dispatches_waits_and_merges() -> None:
    ex = _FakeExecutor(completion=_ok_completion(1))
    outcome = run_node_to_completion(ex, _FakeRalph(completed=True), 1, object(), "main", 30.0)
    assert outcome.success is True
    assert ex.dispatched == [1]
    assert ex.completed == [1]
    assert ex.failed == []


def test_non_completed_run_fails_without_merging() -> None:
    ex = _FakeExecutor()
    outcome = run_node_to_completion(ex, _FakeRalph(completed=False), 2, object(), "main", 30.0)
    assert outcome.success is False
    assert "did not complete" in (outcome.detail or "")
    assert ex.completed == []  # a failed worker must never rebase-merge
    assert ex.failed == [2]


def test_rebase_conflict_is_a_failure_with_detail() -> None:
    conflict = RebaseConflict(
        node_id=3,
        description="x",
        conflicting_files=("a.py", "b.py"),
        detail="CONFLICT in a.py",
    )
    completion = CompletionResult(
        node_id=3, rebased=False, newly_ready=[], rebase_conflict=conflict
    )
    ex = _FakeExecutor(completion=completion)
    outcome = run_node_to_completion(ex, _FakeRalph(completed=True), 3, object(), "main", 30.0)
    assert outcome.success is False
    assert outcome.detail == "CONFLICT in a.py"
    assert ex.completed == [3]
    assert ex.failed == []


def test_completion_timeout_fails_the_node() -> None:
    ex = _FakeExecutor()
    outcome = run_node_to_completion(ex, _FakeRalph(timeout=True), 4, object(), "main", 5.0)
    assert outcome.success is False
    assert "timeout" in (outcome.detail or "")
    assert ex.completed == []
    assert ex.failed == [4]


def test_detached_head_refuses_without_dispatching() -> None:
    """A detached HEAD surfaces as feature_branch == "HEAD"; the loop must refuse
    to dispatch rather than rebase-merge onto the literal ref "HEAD". The node is
    left untouched (pending) for a retry once a real branch is checked out."""
    ex = _FakeExecutor()
    outcome = run_node_to_completion(ex, _FakeRalph(completed=True), 5, object(), "HEAD", 30.0)
    assert outcome.success is False
    assert "HEAD" in (outcome.detail or "")
    assert ex.dispatched == []  # never dispatched onto a detached HEAD
    assert ex.failed == []  # node not force-failed; stays pending for retry
