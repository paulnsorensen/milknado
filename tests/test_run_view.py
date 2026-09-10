from __future__ import annotations

from dataclasses import replace

import pytest

from milknado.app.run import (
    ActiveRunSnapshot,
    ExecutionRunStatus,
    ExecutionSnapshot,
    RunActionAvailability,
    TerminalRunSnapshot,
)
from milknado.app.run_view import (
    actions_text,
    format_attempt,
    format_duration,
    format_eta,
    format_progress,
    output_body,
    output_border_title,
    run_index,
    run_row,
    status_style,
    subtitle_text,
    summary_text,
)


def active_run(**overrides: object) -> ActiveRunSnapshot:
    return replace(
        ActiveRunSnapshot(
            run_id="run-1",
            node_id=3,
            description="Build feature",
            status=ExecutionRunStatus.RUNNING,
            progress="iteration 2",
            stop_requested=False,
            actions=RunActionAvailability(),
            output=(),
            pending_guidance=(),
            elapsed_seconds=125.0,
            progress_pct=55.0,
            eta_seconds=40.0,
            attempt=1,
            max_attempts=3,
            stalled=False,
        ),
        **overrides,
    )


def terminal_run(**overrides: object) -> TerminalRunSnapshot:
    return replace(
        TerminalRunSnapshot(
            run_id="run-2",
            node_id=4,
            description="Ship docs",
            status=ExecutionRunStatus.COMPLETED,
            output=(),
            pending_guidance=(),
            duration_seconds=64.0,
        ),
        **overrides,
    )


def execution_snapshot(**overrides: object) -> ExecutionSnapshot:
    return replace(
        ExecutionSnapshot(
            goal="Ship the release",
            active_runs=(),
            terminal_runs=(),
            completed=3,
            failed=1,
            stopped=2,
            available=5,
            event_lines=(),
        ),
        **overrides,
    )


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (0.0, "00:00"),
        (45.0, "00:45"),
        (59.9, "00:59"),
        (3600.0, "1h00m"),
        (3661.0, "1h01m"),
    ],
)
def test_format_duration(seconds: float, expected: str) -> None:
    assert format_duration(seconds) == expected


def test_format_eta_unknown() -> None:
    assert format_eta(None) == "~?"


def test_format_eta_known_value() -> None:
    assert format_eta(125.0) == "~02:05"


@pytest.mark.parametrize(
    ("pct", "stalled", "expected"),
    [
        (None, False, "—"),
        (None, True, "stalled"),
        (0.0, False, "░" * 10 + " 0%"),
        (55.0, False, "█" * 5 + "░" * 5 + " 55%"),
        (100.0, False, "█" * 10 + " 100%"),
        (150.0, False, "█" * 10 + " 100%"),
        (-20.0, False, "░" * 10 + " 0%"),
    ],
)
def test_format_progress(pct: float | None, stalled: bool, expected: str) -> None:
    assert format_progress(pct, stalled) == expected


def test_format_attempt_first_try_is_blank() -> None:
    assert format_attempt(1, 3) == ""


def test_format_attempt_shows_attempt_over_max() -> None:
    assert format_attempt(2, 3) == "2/3"


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (ExecutionRunStatus.RUNNING, "cyan"),
        (ExecutionRunStatus.COMPLETED, "green"),
        (ExecutionRunStatus.FAILED, "red"),
        (ExecutionRunStatus.STOPPED, "yellow"),
    ],
)
def test_status_style(status: ExecutionRunStatus, expected: str) -> None:
    assert status_style(status) == expected


def test_subtitle_text_renders_available_count() -> None:
    snapshot = execution_snapshot(
        active_runs=(active_run(run_id="run-1"), active_run(run_id="run-2")),
        completed=3,
        failed=1,
        stopped=2,
        available=5,
    )
    assert subtitle_text(snapshot) == (
        "2 active · 3 completed · 1 failed · 2 stopped · 5 available"
    )


def test_summary_text_with_no_run() -> None:
    assert summary_text(None) == "No runs."


def test_summary_text_active_run_renders_elapsed_eta_and_attempt() -> None:
    run = active_run(
        run_id="run-1",
        node_id=3,
        description="Build feature",
        status=ExecutionRunStatus.RUNNING,
        progress="iteration 2",
        stop_requested=True,
        pending_guidance=("check tests", "review"),
        elapsed_seconds=125.0,
        eta_seconds=40.0,
        attempt=2,
        max_attempts=3,
    )
    assert summary_text(run) == (
        "node 3 · run-1\nBuild feature\n"
        "running (stopping) · iteration 2 · elapsed 02:05 · "
        "eta ~00:40 · attempt 2/3\n"
        "Pending guidance: check tests, review"
    )


def test_summary_text_terminal_run_renders_duration() -> None:
    run = terminal_run(
        run_id="run-2",
        node_id=4,
        description="Ship docs",
        status=ExecutionRunStatus.COMPLETED,
        duration_seconds=64.0,
    )
    assert summary_text(run) == (
        "node 4 · run-2\nShip docs\ncompleted · ran 01:04\nUndelivered guidance: none"
    )


def test_actions_text_no_run_selected() -> None:
    assert actions_text(None) == "Actions\nNo run selected."


def test_actions_text_terminal_run_retains_output() -> None:
    assert actions_text(terminal_run()) == (
        "Actions\nRun has stopped; output retained for inspection."
    )


def test_actions_text_active_run_all_actions_available() -> None:
    assert actions_text(active_run(actions=RunActionAvailability())) == (
        "Actions\nAll actions available."
    )


def test_actions_text_active_run_lists_blocked_reasons() -> None:
    run = active_run(
        actions=RunActionAvailability(
            cancel_reason="Cannot cancel.",
            force_stop_reason="No child process is active.",
        )
    )
    assert actions_text(run) == (
        "Actions\nCancel: Cannot cancel.\nForce stop: No child process is active."
    )


def test_output_border_title_following() -> None:
    assert output_border_title(auto_follow=True) == "Output (following newest output)"


def test_output_border_title_paused() -> None:
    assert output_border_title(auto_follow=False) == "Output (paused; press r to resume)"


def test_output_body_empty() -> None:
    assert output_body(None) == "No output yet."
    assert output_body(active_run(output=())) == "No output yet."


def test_output_body_non_empty() -> None:
    run = active_run(output=("line1", "line2"))
    assert output_body(run) == "line1\nline2"


def test_run_row_active_includes_retry_count() -> None:
    run = active_run(
        run_id="run-1",
        node_id=3,
        description="Build feature",
        status=ExecutionRunStatus.RUNNING,
        elapsed_seconds=125.0,
        progress_pct=55.0,
        stalled=False,
        attempt=2,
        max_attempts=3,
    )
    node_id, description, status_cell, progress, elapsed = run_row(run)

    assert node_id == "3"
    assert description == "Build feature"
    assert status_cell.plain == "running 2/3"
    assert progress == "█" * 5 + "░" * 5 + " 55%"
    assert elapsed == "02:05"


def test_run_row_active_without_retry_has_clean_status_text() -> None:
    run = active_run(
        status=ExecutionRunStatus.RUNNING,
        attempt=1,
        max_attempts=3,
    )
    _, _, status_cell, _, _ = run_row(run)

    assert status_cell.plain == "running"


def test_run_row_terminal() -> None:
    run = terminal_run(
        run_id="run-2",
        node_id=4,
        description="Ship docs",
        status=ExecutionRunStatus.COMPLETED,
        duration_seconds=64.0,
    )
    node_id, description, status_cell, progress, elapsed = run_row(run)

    assert node_id == "4"
    assert description == "Ship docs"
    assert status_cell.plain == "completed"
    assert progress == "—"
    assert elapsed == "01:04"


def test_run_index_selected_present() -> None:
    runs = (active_run(run_id="run-1"), active_run(run_id="run-2"))
    assert run_index(runs, "run-2") == 1


def test_run_index_selected_absent() -> None:
    runs = (active_run(run_id="run-1"), active_run(run_id="run-2"))
    assert run_index(runs, "missing") == 0


def test_run_index_empty_runs() -> None:
    assert run_index((), None) == 0
