from __future__ import annotations

from pathlib import Path
from typing import Literal

import milknado.domains.execution.headless as headless
from milknado.domains.common import ProgressEvent
from milknado.domains.common.errors import CompletionTimeout
from milknado.domains.execution import HeadlessOutcome, run_node_to_completion
from milknado.domains.execution.executor import (
    CompletionResult,
    DispatchResult,
    ExecutionConfig,
    RebaseConflict,
)

_EXEC_CONFIG = ExecutionConfig(
    execution_agent="agent",
    quality_gates=(),
    worktree_pattern="wt-{node_id}",
    project_root=Path("/tmp"),
)


class _FakeExecutor:
    def __init__(self, completion: CompletionResult | None = None) -> None:
        self._completion = completion
        self.dispatched: list[int] = []
        self.completed: list[int] = []
        self.cancelled: list[int] = []
        self.failed: list[int] = []
        self.unconfirmed_stops: list[str] = []

    def dispatch(
        self,
        node_id: int,
        config: ExecutionConfig,
        *,
        base_oid: str | None = None,
    ) -> DispatchResult:
        assert isinstance(config, ExecutionConfig)
        self.dispatched.append(node_id)
        return DispatchResult(node_id=node_id, worktree=Path("/tmp/wt"), run_id=f"run-{node_id}")

    def complete(self, node_id: int, feature_branch: str) -> CompletionResult:
        self.completed.append(node_id)
        assert self._completion is not None
        return self._completion

    def fail(self, node_id: int) -> None:
        self.failed.append(node_id)

    def cancel(self, node_id: int) -> None:
        self.cancelled.append(node_id)

    def note_unconfirmed_stop(self, run_id: str) -> None:
        self.unconfirmed_stops.append(run_id)


class _FakeRalph:
    def __init__(
        self,
        *,
        outcome: Literal["completed", "stopped", "failed"] = "completed",
        timeout: bool = False,
    ) -> None:
        self._outcome = outcome
        self._timeout = timeout
        self.stopped: list[str] = []

    def wait_for_next_completion(self, active_run_ids, timeout=None):
        if self._timeout:
            raise CompletionTimeout(active_run_ids=active_run_ids, waited_seconds=timeout or 0.0)
        return next(iter(active_run_ids)), self._outcome

    def stop_run(self, run_id: str, timeout: float | None = None) -> bool:
        self.stopped.append(run_id)
        return True


class _ProgressThenTerminalRalph(_FakeRalph):
    def __init__(self) -> None:
        super().__init__()
        self._outcomes = iter(
            (
                ProgressEvent(run_id="run-13", work=1, total=0, message="iteration 1 started"),
                "completed",
            )
        )
        self.waits = 0
        self.timeouts: list[float | None] = []

    def wait_for_next_completion(self, active_run_ids, timeout=None):
        self.timeouts.append(timeout)
        self.waits += 1
        return next(iter(active_run_ids)), next(self._outcomes)


def _ok_completion(node_id: int) -> CompletionResult:
    return CompletionResult(node_id=node_id, rebased=True, newly_ready=[], rebase_conflict=None)


def test_success_dispatches_waits_and_merges() -> None:
    ex = _FakeExecutor(completion=_ok_completion(1))
    outcome = run_node_to_completion(
        ex, _FakeRalph(outcome="completed"), 1, _EXEC_CONFIG, "main", 30.0
    )
    assert outcome.success is True
    assert ex.dispatched == [1]
    assert ex.completed == [1]
    assert ex.failed == []


def test_non_completed_run_fails_without_merging() -> None:
    ex = _FakeExecutor()
    outcome = run_node_to_completion(
        ex, _FakeRalph(outcome="failed"), 2, _EXEC_CONFIG, "main", 30.0
    )
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
    outcome = run_node_to_completion(
        ex, _FakeRalph(outcome="completed"), 3, _EXEC_CONFIG, "main", 30.0
    )
    assert outcome.success is False
    assert outcome.detail == "CONFLICT in a.py"
    assert ex.completed == [3]
    assert ex.failed == []


def test_completion_timeout_fails_the_node() -> None:
    ex = _FakeExecutor()
    outcome = run_node_to_completion(ex, _FakeRalph(timeout=True), 4, _EXEC_CONFIG, "main", 5.0)
    assert outcome.success is False
    assert "timeout" in (outcome.detail or "")
    assert ex.completed == []
    assert ex.failed == [4]


def test_detached_head_refuses_without_dispatching() -> None:
    """A detached HEAD surfaces as feature_branch == "HEAD"; the loop must refuse
    to dispatch rather than rebase-merge onto the literal ref "HEAD". The node is
    marked failed for parity with the other failure branches (reset to pending to
    retry once a real branch is checked out)."""
    ex = _FakeExecutor()
    outcome = run_node_to_completion(
        ex, _FakeRalph(outcome="completed"), 5, _EXEC_CONFIG, "HEAD", 30.0
    )
    assert outcome.success is False
    assert "HEAD" in (outcome.detail or "")
    assert ex.dispatched == []  # never dispatched onto a detached HEAD
    assert ex.failed == [5]  # marked failed for parity with sibling failure paths


def test_timeout_stops_the_ralph_run() -> None:
    """#46: CompletionTimeout must stop the underlying ralph run so the loop does
    not keep running as a zombie after the timeout fires."""
    ex = _FakeExecutor()
    ralph = _FakeRalph(timeout=True)
    outcome = run_node_to_completion(ex, ralph, 10, _EXEC_CONFIG, "main", 5.0)
    assert outcome.success is False
    assert "timeout" in (outcome.detail or "")
    assert ralph.stopped == ["run-10"], "timeout must stop the ralph run"


def test_non_completed_stops_the_ralph_run() -> None:
    """#56: A non-completed run must also stop the ralph loop before failing the
    node, same class of fix as #46."""
    ex = _FakeExecutor()
    ralph = _FakeRalph(outcome="failed")
    outcome = run_node_to_completion(ex, ralph, 11, _EXEC_CONFIG, "main", 30.0)
    assert outcome.success is False
    assert ralph.stopped == ["run-11"], "non-completion must stop the ralph run"


def test_stopped_run_cancels_without_merging() -> None:
    ex = _FakeExecutor()
    outcome = run_node_to_completion(
        ex, _FakeRalph(outcome="stopped"), 12, _EXEC_CONFIG, "main", 30.0
    )

    assert outcome == HeadlessOutcome(12, success=False, detail="worker run stopped")
    assert ex.completed == []
    assert ex.cancelled == [12]
    assert ex.failed == []


def test_progress_event_waits_for_terminal_outcome_before_merging() -> None:
    ex = _FakeExecutor(completion=_ok_completion(13))
    ralph = _ProgressThenTerminalRalph()

    outcome = run_node_to_completion(ex, ralph, 13, _EXEC_CONFIG, "main", 30.0)

    assert outcome.success is True
    assert ralph.waits == 2
    assert ex.completed == [13]


def test_progress_event_uses_remaining_completion_deadline(
    monkeypatch,
) -> None:
    ex = _FakeExecutor(completion=_ok_completion(13))
    ralph = _ProgressThenTerminalRalph()
    monotonic = iter((100.0, 101.0, 106.0))
    monkeypatch.setattr(headless.time, "monotonic", lambda: next(monotonic))

    outcome = run_node_to_completion(ex, ralph, 13, _EXEC_CONFIG, "main", 30.0)

    assert outcome.success is True
    assert ralph.timeouts == [29.0, 24.0]


def test_timeout_preserves_ownership_when_worker_does_not_exit() -> None:
    """High: an unconfirmed stop here must register with the executor's
    cancel watcher (note_unconfirmed_stop) — headless.py issues its own
    stop_run outside Executor's own abort paths, so without this the row
    would never be finalized once the wedged loop later self-exits."""
    ex = _FakeExecutor()
    ralph = _FakeRalph(timeout=True)
    ralph.stop_run = lambda run_id, timeout=None: False

    result = run_node_to_completion(ex, ralph, 1, _EXEC_CONFIG, "main", 0.01)

    assert result.success is False
    assert result.detail == "completion timeout; worker did not exit, ownership preserved"
    assert ex.failed == []
    assert ex.unconfirmed_stops == ["run-1"]


def test_incomplete_run_preserves_ownership_when_worker_does_not_exit() -> None:
    """High: same note_unconfirmed_stop registration as the timeout branch
    above — the non-completed-run unconfirmed-stop path had the same gap."""
    ex = _FakeExecutor()
    ralph = _FakeRalph(outcome="failed")
    ralph.stop_run = lambda run_id, timeout=None: False

    result = run_node_to_completion(ex, ralph, 1, _EXEC_CONFIG, "main", 0.01)

    assert result.success is False
    assert result.detail == "worker run did not complete or exit; ownership preserved"
    assert ex.failed == []
    assert ex.unconfirmed_stops == ["run-1"]
