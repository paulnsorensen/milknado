from __future__ import annotations

from pathlib import Path

from milknado.domains.common.errors import (
    CompletionTimeout,
    ExistingPlanDetected,
    InsufficientTestCoverageError,
    InvalidTransition,
    MegaBatchAborted,
    MilknadoError,
    MultiStoryBundlingError,
    PlanningFailed,
    RalphMarkdownWriteError,
    RebaseAbortError,
    TransientDispatchError,
)
from milknado.domains.common.types import NodeStatus


class TestMilknadoError:
    def test_is_exception(self) -> None:
        err = MilknadoError("base error")
        assert isinstance(err, Exception)
        assert str(err) == "base error"


class TestRebaseAbortError:
    def test_message_includes_worktree(self) -> None:
        err = RebaseAbortError(Path("/some/worktree"))
        assert "/some/worktree" in str(err)
        assert err.worktree == Path("/some/worktree")

    def test_stores_stderr(self) -> None:
        err = RebaseAbortError(Path("/wt"), stderr="fatal: rebase is running")
        assert err.stderr == "fatal: rebase is running"

    def test_default_stderr_is_empty(self) -> None:
        err = RebaseAbortError(Path("/wt"))
        assert err.stderr == ""

    def test_is_milknado_error(self) -> None:
        assert isinstance(RebaseAbortError(Path("/wt")), MilknadoError)


class TestRalphMarkdownWriteError:
    def test_message_includes_path(self) -> None:
        err = RalphMarkdownWriteError(Path("/project/RALPH.md"))
        assert "/project/RALPH.md" in str(err)

    def test_stores_cause(self) -> None:
        cause = OSError("disk full")
        err = RalphMarkdownWriteError(Path("/p"), cause=cause)
        assert err.cause is cause

    def test_default_cause_is_none(self) -> None:
        err = RalphMarkdownWriteError(Path("/p"))
        assert err.cause is None


class TestCompletionTimeout:
    def test_message_includes_run_ids(self) -> None:
        err = CompletionTimeout(active_run_ids={"run-1", "run-2"}, waited_seconds=30.5)
        msg = str(err)
        assert "run-1" in msg
        assert "run-2" in msg
        assert "30.5" in msg

    def test_stores_attributes(self) -> None:
        ids = {"run-abc"}
        err = CompletionTimeout(active_run_ids=ids, waited_seconds=60.0)
        assert err.active_run_ids == ids
        assert err.waited_seconds == 60.0

    def test_default_waited_is_zero(self) -> None:
        err = CompletionTimeout(active_run_ids=set())
        assert err.waited_seconds == 0.0


class TestPlanningFailed:
    def test_message_includes_exit_code_and_stderr(self) -> None:
        err = PlanningFailed(exit_code=2, stderr="something went wrong")
        msg = str(err)
        assert "2" in msg
        assert "something went wrong" in msg

    def test_truncates_long_stderr(self) -> None:
        long_stderr = "x" * 500
        err = PlanningFailed(exit_code=1, stderr=long_stderr)
        assert len(str(err)) < 600


class TestInvalidTransition:
    def test_message_includes_states(self) -> None:
        err = InvalidTransition(
            node_id=5,
            current=NodeStatus.PENDING,
            target=NodeStatus.DONE,
            valid_targets=(NodeStatus.RUNNING, NodeStatus.BLOCKED),
        )
        msg = str(err)
        assert "5" in msg
        assert "pending" in msg
        assert "done" in msg

    def test_is_value_error(self) -> None:
        err = InvalidTransition(
            node_id=1,
            current=NodeStatus.DONE,
            target=NodeStatus.PENDING,
            valid_targets=(),
        )
        assert isinstance(err, ValueError)

    def test_stores_attributes(self) -> None:
        err = InvalidTransition(
            node_id=3,
            current=NodeStatus.RUNNING,
            target=NodeStatus.FAILED,
            valid_targets=(NodeStatus.DONE,),
        )
        assert err.node_id == 3
        assert err.current == NodeStatus.RUNNING
        assert err.target == NodeStatus.FAILED


class TestTransientDispatchError:
    def test_is_milknado_error(self) -> None:
        err = TransientDispatchError("transient")
        assert isinstance(err, MilknadoError)


class TestMegaBatchAborted:
    def test_message_includes_counts(self) -> None:
        err = MegaBatchAborted(change_count=50, threshold=20)
        msg = str(err)
        assert "50" in msg
        assert "20" in msg

    def test_stores_attributes(self) -> None:
        err = MegaBatchAborted(change_count=10, threshold=5)
        assert err.change_count == 10
        assert err.threshold == 5


class TestExistingPlanDetected:
    def test_message_includes_counts(self) -> None:
        err = ExistingPlanDetected(total=10, done=3, pending=5, running=2)
        msg = str(err)
        assert "10" in msg
        assert "3" in msg
        assert "5" in msg
        assert "2" in msg

    def test_stores_attributes(self) -> None:
        err = ExistingPlanDetected(total=4, done=1, pending=2, running=1)
        assert err.total == 4
        assert err.done == 1
        assert err.pending == 2
        assert err.running == 1


class TestMultiStoryBundlingError:
    def test_message_includes_count(self) -> None:
        err = MultiStoryBundlingError(bundled_changes=["a", "b", "c"])
        assert "3" in str(err)

    def test_stores_changes(self) -> None:
        changes = ["change-1", "change-2"]
        err = MultiStoryBundlingError(bundled_changes=changes)
        assert err.bundled_changes == changes


class TestInsufficientTestCoverageError:
    def test_message_includes_count(self) -> None:
        err = InsufficientTestCoverageError(orphan_changes=["impl-1", "impl-2"])
        assert "2" in str(err)
        assert "impl-1" in str(err)

    def test_stores_changes(self) -> None:
        changes = ["impl-x"]
        err = InsufficientTestCoverageError(orphan_changes=changes)
        assert err.orphan_changes == changes
