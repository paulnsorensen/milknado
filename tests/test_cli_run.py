"""Tests for cli_run.py presentation + resume helpers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from milknado.cli_run import _print_run_result, _reconcile_stale_if_resume
from milknado.domains.execution.executor import RebaseConflict
from milknado.domains.execution.pr_stack import StackedPr
from milknado.domains.execution.run_loop._result import RunLoopResult


def _printed(console: MagicMock) -> str:
    return "\n".join(str(call.args[0]) for call in console.print.call_args_list if call.args)


def test_print_run_result_root_done_reports_success() -> None:
    result = RunLoopResult(root_done=True, dispatched_total=3, completed_total=3, failed_total=0)
    with patch("milknado.cli_run.console", MagicMock()) as console:
        _print_run_result(result)
    assert "Root goal achieved" in _printed(console)


def test_print_run_result_incomplete_reports_counts() -> None:
    result = RunLoopResult(root_done=False, dispatched_total=4, completed_total=2, failed_total=1)
    with patch("milknado.cli_run.console", MagicMock()) as console:
        _print_run_result(result)
    out = _printed(console)
    assert "2 completed" in out
    assert "1 failed" in out


def test_print_run_result_lists_stacked_prs() -> None:
    result = RunLoopResult(
        root_done=True,
        dispatched_total=1,
        completed_total=1,
        failed_total=0,
        stacked_prs=(
            StackedPr(
                node_id=7,
                branch="node-7",
                base_branch="main",
                pr_url="https://gh/pr/7",
            ),
        ),
    )
    with patch("milknado.cli_run.console", MagicMock()) as console:
        _print_run_result(result)
    out = _printed(console)
    assert "node 7" in out
    assert "node-7 → main" in out
    assert "https://gh/pr/7" in out


def test_print_run_result_renders_rebase_conflict_with_files_and_detail() -> None:
    result = RunLoopResult(
        root_done=False,
        dispatched_total=1,
        completed_total=0,
        failed_total=1,
        rebase_conflicts=(
            RebaseConflict(
                node_id=9,
                description="overlapping edit",
                conflicting_files=("src/a.py", "src/b.py"),
                detail="rerere could not resolve",
            ),
        ),
    )
    with patch("milknado.cli_run.console", MagicMock()) as console:
        _print_run_result(result)
    out = _printed(console)
    assert "node 9" in out
    assert "overlapping edit" in out
    assert "src/a.py" in out
    assert "src/b.py" in out
    assert "rerere could not resolve" in out


def test_reconcile_stale_if_resume_returns_none_when_not_resuming() -> None:
    git = MagicMock()
    assert _reconcile_stale_if_resume(MagicMock(), git, MagicMock(), resume=False) is None
    git.current_branch.assert_not_called()


def test_reconcile_stale_if_resume_reconciles_and_returns_branch() -> None:
    git = MagicMock()
    git.current_branch.return_value = "feature-x"
    reconcile_result = MagicMock(resolved_done=[1, 2], reset_pending=[3])
    with (
        patch(
            "milknado.domains.execution.reconcile_stale_nodes",
            return_value=reconcile_result,
        ) as reconcile,
        patch("milknado.cli_run.console", MagicMock()) as console,
    ):
        branch = _reconcile_stale_if_resume(MagicMock(), git, MagicMock(), resume=True)
    assert branch == "feature-x"
    reconcile.assert_called_once()
    assert "Reconciled" in _printed(console)


def test_reconcile_stale_if_resume_quiet_when_nothing_to_reconcile() -> None:
    git = MagicMock()
    git.current_branch.return_value = "feature-y"
    reconcile_result = MagicMock(resolved_done=[], reset_pending=[])
    with (
        patch(
            "milknado.domains.execution.reconcile_stale_nodes",
            return_value=reconcile_result,
        ),
        patch("milknado.cli_run.console", MagicMock()) as console,
    ):
        branch = _reconcile_stale_if_resume(MagicMock(), git, MagicMock(), resume=True)
    assert branch == "feature-y"
    assert "Reconciled" not in _printed(console)
