from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import chain, repeat
from pathlib import Path
from typing import Literal, cast

import pytest
from typing_extensions import override

from milknado.domains.common import ProgressEvent, TerminalRunOutcome
from milknado.domains.common.errors import CompletionTimeout
from milknado.domains.common.protocols import LoopPort
from milknado.domains.execution import Executor, NodeLoopOutcome, RunLoop
from milknado.domains.execution._models import (
    CompletionResult,
    DispatchResult,
    PreservedWorkerRun,
    RebaseConflict,
)
from milknado.domains.execution.executor import ExecutionConfig
from milknado.domains.graph import MikadoGraph

_EXEC_CONFIG = ExecutionConfig(
    execution_agent="agent",
    quality_gates=(),
    worktree_pattern="wt-{node_id}",
    project_root=Path("/tmp"),
)


class _FakeExecutor:
    _completion: CompletionResult | None

    def __init__(self, completion: CompletionResult | None = None) -> None:
        self._completion = completion
        self.dispatched: list[int] = []
        self.completed: list[int] = []
        self.cancelled: list[int] = []
        self.failed: list[int] = []
        self.stopped: list[str] = []
        self.force_stopped: list[str] = []
        self.stop_result: bool = True

    def dispatch(
        self,
        node_id: int,
        config: ExecutionConfig,
        *,
        base_oid: str | None = None,
        parent_run_id: str | None = None,
    ) -> DispatchResult:
        _ = (base_oid, parent_run_id)
        assert isinstance(config, ExecutionConfig)
        self.dispatched.append(node_id)
        return DispatchResult(node_id=node_id, worktree=Path("/tmp/wt"), run_id=f"run-{node_id}")

    def complete(self, node_id: int, feature_branch: str) -> CompletionResult:
        _ = feature_branch
        self.completed.append(node_id)
        assert self._completion is not None
        return self._completion

    def fail(self, node_id: int, detail: str | None = None) -> None:
        _ = detail
        self.failed.append(node_id)

    def cancel(self, node_id: int) -> None:
        self.cancelled.append(node_id)

    def stop_run(self, run_id: str, timeout: float | None = None) -> bool:
        _ = timeout
        self.stopped.append(run_id)
        return self.stop_result

    def force_stop_run(self, run_id: str, timeout: float | None = None) -> bool:
        _ = timeout
        self.force_stopped.append(run_id)
        return self.stop_result


class _FakeLoop:
    outcome: TerminalRunOutcome
    _timeout: bool
    stop_result: bool

    def __init__(
        self,
        *,
        outcome: Literal["completed", "stopped", "failed"] = "completed",
        timeout: bool = False,
    ) -> None:
        self.outcome = TerminalRunOutcome(outcome)
        self._timeout = timeout
        self.stop_result = True
        self.stopped: list[str] = []

    def wait_for_next_completion(
        self, active_run_ids: set[str], timeout: float | None = None
    ) -> tuple[str, TerminalRunOutcome | ProgressEvent]:
        if self._timeout:
            raise CompletionTimeout(active_run_ids=active_run_ids, waited_seconds=timeout or 0.0)
        return next(iter(active_run_ids)), self.outcome

    def stop_run(self, run_id: str, timeout: float | None = None) -> bool:
        _ = timeout
        self.stopped.append(run_id)
        return self.stop_result

    def get_run_output_tail(self, run_id: str, lines: int) -> list[str]:
        _ = (run_id, lines)
        return []

    def get_run_guidance(self, run_id: str) -> list[str]:
        _ = run_id
        return []

    def get_run_session(self, run_id: str) -> None:
        _ = run_id
        return None

    def get_run_failure_detail(self, run_id: str) -> None:
        _ = run_id
        return None


class _ProgressThenTerminalLoop(_FakeLoop):
    _outcomes: Iterator[ProgressEvent | TerminalRunOutcome]

    def __init__(self) -> None:
        super().__init__()
        self._outcomes = iter(
            (
                ProgressEvent(run_id="run-13", work=1, total=0, message="iteration 1 started"),
                TerminalRunOutcome("completed"),
            )
        )
        self.waits: int = 0
        self.timeouts: list[float | None] = []

    @override
    def wait_for_next_completion(
        self, active_run_ids: set[str], timeout: float | None = None
    ) -> tuple[str, TerminalRunOutcome | ProgressEvent]:
        self.timeouts.append(timeout)
        self.waits += 1
        return next(iter(active_run_ids)), next(self._outcomes)


class _Graph:
    def get_node(self, node_id: int) -> None:
        _ = node_id
        return None


@dataclass(frozen=True)
class _NodeRun:
    config: ExecutionConfig
    feature_branch: str
    timeout: float


def run_node_to_completion(
    executor: _FakeExecutor, loop: _FakeLoop, node_id: int, run: _NodeRun
) -> NodeLoopOutcome:
    driver = RunLoop(
        executor=cast(Executor, cast(object, executor)),
        graph=cast(MikadoGraph, cast(object, _Graph())),
        loop=cast(LoopPort, cast(object, loop)),
    )
    return driver.run_node(node_id, run.config, run.feature_branch, run.timeout)


def _ok_completion(node_id: int) -> CompletionResult:
    return CompletionResult(node_id=node_id, rebased=True, newly_ready=[], rebase_conflict=None)


def test_success_dispatches_waits_and_merges() -> None:
    ex = _FakeExecutor(completion=_ok_completion(1))
    outcome = run_node_to_completion(
        ex, _FakeLoop(outcome="completed"), 1, _NodeRun(_EXEC_CONFIG, "main", 30.0)
    )
    assert outcome.success is True
    assert ex.dispatched == [1]
    assert ex.completed == [1]
    assert ex.failed == []


def test_non_completed_run_fails_without_merging() -> None:
    ex = _FakeExecutor()
    outcome = run_node_to_completion(
        ex, _FakeLoop(outcome="failed"), 2, _NodeRun(_EXEC_CONFIG, "main", 30.0)
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
        ex, _FakeLoop(outcome="completed"), 3, _NodeRun(_EXEC_CONFIG, "main", 30.0)
    )
    assert outcome.success is False
    assert outcome.detail == "CONFLICT in a.py"
    assert ex.completed == [3]
    assert ex.failed == []


def test_completion_timeout_fails_the_node() -> None:
    ex = _FakeExecutor()
    outcome = run_node_to_completion(
        ex, _FakeLoop(timeout=True), 4, _NodeRun(_EXEC_CONFIG, "main", 5.0)
    )
    assert outcome.success is False
    assert outcome.timed_out is True
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
        ex, _FakeLoop(outcome="completed"), 5, _NodeRun(_EXEC_CONFIG, "HEAD", 30.0)
    )
    assert outcome.success is False
    assert "HEAD" in (outcome.detail or "")
    assert ex.dispatched == []  # never dispatched onto a detached HEAD
    assert ex.failed == [5]  # marked failed for parity with sibling failure paths


def test_timeout_stops_the_loop_run() -> None:
    """#46: CompletionTimeout must stop the underlying loop run so the loop does
    not keep running as a zombie after the timeout fires."""
    ex = _FakeExecutor()
    loop = _FakeLoop(timeout=True)
    outcome = run_node_to_completion(ex, loop, 10, _NodeRun(_EXEC_CONFIG, "main", 5.0))
    assert outcome.success is False
    assert "timeout" in (outcome.detail or "")
    assert ex.force_stopped == ["run-10"], "executor must force-stop the loop run"


def test_failed_timeout_outcome_sets_headless_timeout() -> None:
    ex = _FakeExecutor()
    loop = _FakeLoop(outcome="failed")
    loop.outcome = TerminalRunOutcome("failed", timed_out=True)

    outcome = run_node_to_completion(ex, loop, 10, _NodeRun(_EXEC_CONFIG, "main", 30.0))

    assert outcome.success is False
    assert outcome.timed_out is True


def test_failed_timeout_preserves_timeout_when_stop_fails() -> None:
    ex = _FakeExecutor()
    ex.stop_result = False
    loop = _FakeLoop(outcome="failed")
    loop.outcome = TerminalRunOutcome("failed", timed_out=True)

    outcome = run_node_to_completion(ex, loop, 10, _NodeRun(_EXEC_CONFIG, "main", 30.0))

    assert outcome.success is False
    assert outcome.timed_out is True


def test_non_completed_stops_the_loop_run() -> None:
    """#56: A non-completed run must also stop the loop before failing the
    node, same class of fix as #46."""
    ex = _FakeExecutor()
    loop = _FakeLoop(outcome="failed")
    outcome = run_node_to_completion(ex, loop, 11, _NodeRun(_EXEC_CONFIG, "main", 30.0))
    assert outcome.success is False
    assert ex.stopped == ["run-11"], "executor must stop the loop run"


def test_stopped_run_cancels_without_merging() -> None:
    ex = _FakeExecutor()
    outcome = run_node_to_completion(
        ex, _FakeLoop(outcome="stopped"), 12, _NodeRun(_EXEC_CONFIG, "main", 30.0)
    )

    assert outcome == NodeLoopOutcome(12, success=False, detail="worker run stopped")
    assert ex.completed == []
    assert ex.cancelled == [12]
    assert ex.failed == []


def test_redispatch_keeps_the_aggregate_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dataclasses import replace

    class _RedispatchExecutor(_FakeExecutor):
        def __init__(self) -> None:
            super().__init__()
            self.completions: int = 0

        @override
        def complete(self, node_id: int, feature_branch: str) -> CompletionResult:
            self.completions += 1
            if self.completions == 1:
                return CompletionResult(
                    node_id=node_id,
                    rebased=False,
                    newly_ready=[],
                    redispatch=DispatchResult(
                        node_id=node_id, worktree=Path("/tmp/wt-2"), run_id="run-2"
                    ),
                )
            return super().complete(node_id, feature_branch)

    class _RedispatchLoop(_FakeLoop):
        def __init__(self) -> None:
            super().__init__()
            self.timeouts: list[float | None] = []
            self.waits: int = 0

        @override
        def wait_for_next_completion(
            self, active_run_ids: set[str], timeout: float | None = None
        ) -> tuple[str, TerminalRunOutcome | ProgressEvent]:
            self.timeouts.append(timeout)
            self.waits += 1
            if self.waits == 1:
                return next(iter(active_run_ids)), TerminalRunOutcome("completed")
            raise CompletionTimeout(active_run_ids=active_run_ids, waited_seconds=timeout or 0.0)

    monotonic = chain((100.0, 101.0, 102.0), repeat(102.0))
    monkeypatch.setattr(time, "monotonic", lambda: next(monotonic))
    executor = _RedispatchExecutor()
    loop = _RedispatchLoop()
    config = replace(_EXEC_CONFIG, max_iterations=2)

    result = run_node_to_completion(executor, loop, 14, _NodeRun(config, "main", 3.0))

    assert result.timed_out is True
    assert loop.timeouts == [5.0, 4.0]


def test_progress_event_waits_for_terminal_outcome_before_merging() -> None:
    ex = _FakeExecutor(completion=_ok_completion(13))
    loop = _ProgressThenTerminalLoop()

    outcome = run_node_to_completion(ex, loop, 13, _NodeRun(_EXEC_CONFIG, "main", 30.0))

    assert outcome.success is True
    assert loop.waits == 2
    assert ex.completed == [13]


def test_progress_event_uses_remaining_completion_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ex = _FakeExecutor(completion=_ok_completion(13))
    loop = _ProgressThenTerminalLoop()
    monotonic = chain((100.0, 101.0, 106.0), repeat(106.0))
    monkeypatch.setattr(time, "monotonic", lambda: next(monotonic))

    outcome = run_node_to_completion(ex, loop, 13, _NodeRun(_EXEC_CONFIG, "main", 30.0))

    assert outcome.success is True
    assert loop.timeouts == [29.0, 24.0]


def test_timeout_preserves_ownership_when_worker_does_not_exit() -> None:
    """An unconfirmed stop stays owned by the executor for watcher cleanup."""
    ex = _FakeExecutor()
    loop = _FakeLoop(timeout=True)
    ex.stop_result = False

    result = run_node_to_completion(ex, loop, 1, _NodeRun(_EXEC_CONFIG, "main", 0.01))

    assert result.success is False
    assert result.detail == "completion timeout; worker did not exit, ownership preserved"
    assert result.ownership_preserved is True
    assert ex.failed == []
    assert ex.force_stopped == ["run-1"]


def test_incomplete_run_preserves_ownership_when_worker_does_not_exit() -> None:
    """An incomplete run uses the executor stop wrapper before returning."""
    ex = _FakeExecutor()
    loop = _FakeLoop(outcome="failed")
    ex.stop_result = False

    result = run_node_to_completion(ex, loop, 1, _NodeRun(_EXEC_CONFIG, "main", 0.01))

    assert result.success is False
    assert result.detail == "worker run did not complete or exit; ownership preserved"
    assert result.ownership_preserved is True
    assert ex.failed == []
    assert ex.stopped == ["run-1"]


def test_detached_supervision_waits_for_confirmed_worker_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _RecoveringExecutor(_FakeExecutor):
        @override
        def force_stop_run(self, run_id: str, timeout: float | None = None) -> bool:
            _ = timeout
            self.force_stopped.append(run_id)
            return len(self.force_stopped) == 2

    executor = _RecoveringExecutor()
    executor.stop_result = False
    loop = _FakeLoop(outcome="failed")
    driver = RunLoop(
        executor=cast(Executor, cast(object, executor)),
        graph=cast(MikadoGraph, cast(object, _Graph())),
        loop=cast(LoopPort, cast(object, loop)),
    )
    pending = driver.run_node(1, _EXEC_CONFIG, "main", 30.0)
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        assert executor.failed == []
        sleeps.append(seconds)

    monkeypatch.setattr(time, "sleep", sleep)
    resolved = driver.confirm_preserved_stop(pending)

    assert pending.ownership_preserved is True
    assert resolved.ownership_preserved is False
    assert executor.force_stopped == ["run-1", "run-1"]
    assert executor.failed == [1]
    assert sleeps == [1.0]


def test_post_start_dispatch_error_keeps_detached_worker_owned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _AbortedExecutor(_FakeExecutor):
        @override
        def dispatch(
            self,
            node_id: int,
            config: ExecutionConfig,
            *,
            base_oid: str | None = None,
            parent_run_id: str | None = None,
        ) -> DispatchResult:
            raise PreservedWorkerRun(node_id, "started-run")

        @override
        def force_stop_run(self, run_id: str, timeout: float | None = None) -> bool:
            self.force_stopped.append(run_id)
            return len(self.force_stopped) == 2

    executor = _AbortedExecutor()
    driver = RunLoop(
        executor=cast(Executor, cast(object, executor)),
        graph=cast(MikadoGraph, cast(object, _Graph())),
        loop=cast(LoopPort, cast(object, _FakeLoop())),
    )
    pending = driver.run_node(1, _EXEC_CONFIG, "main", 30.0)
    assert pending.ownership_preserved is True
    assert executor.failed == []

    def no_sleep(_seconds: float) -> None:
        pass

    monkeypatch.setattr(time, "sleep", no_sleep)
    resolved = driver.confirm_preserved_stop(pending)
    assert resolved.ownership_preserved is False
    assert executor.force_stopped == ["started-run", "started-run"]
    assert executor.failed == [1]
