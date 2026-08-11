from __future__ import annotations

import pytest
from rich.text import Text

from milknado.app.run import (
    ActiveRunSnapshot,
    ExecutionRunStatus,
    ExecutionSnapshot,
    RunActionAvailability,
    TerminalRunSnapshot,
)
from milknado.app.run_view import (
    actions_text,
    confirmation_text,
    events_text,
    format_attempt,
    format_duration,
    format_eta,
    format_progress,
    help_text,
    output_body,
    output_border_title,
    run_index,
    run_row,
    status_style,
    subtitle_text,
    summary_text,
)


def active_run(**overrides: object) -> ActiveRunSnapshot:
    defaults: dict[str, object] = dict(
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
    )
    defaults.update(overrides)
    return ActiveRunSnapshot(**defaults)


def terminal_run(**overrides: object) -> TerminalRunSnapshot:
    defaults: dict[str, object] = dict(
        run_id="run-2",
        node_id=4,
        description="Ship docs",
        status=ExecutionRunStatus.COMPLETED,
        output=(),
        pending_guidance=(),
        duration_seconds=64.0,
    )
    defaults.update(overrides)
    return TerminalRunSnapshot(**defaults)


def execution_snapshot(**overrides: object) -> ExecutionSnapshot:
    defaults: dict[str, object] = dict(
        goal="Ship the release",
        active_runs=(),
        terminal_runs=(),
        completed=3,
        failed=1,
        stopped=2,
        available=5,
        event_lines=(),
    )
    defaults.update(overrides)
    return ExecutionSnapshot(**defaults)


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


def test_help_text_compact_list_route() -> None:
    assert help_text(None, compact=True, route="list", auto_follow=True) == (
        "Help\n↑/↓ select · h close help · q quit · enter open"
    )


def test_help_text_compact_detail_route() -> None:
    assert help_text(None, compact=True, route="detail", auto_follow=True) == (
        "Help\n↑/↓ select · h close help · q quit · escape back"
    )


def test_help_text_wide_shows_available_run_actions() -> None:
    run = active_run(actions=RunActionAvailability())
    assert help_text(run, compact=False, route="list", auto_follow=True) == (
        "Help\n↑/↓ select · h close help · q quit · g queue guidance · c cancel · f force stop"
    )


def test_help_text_paused_output_offers_resume() -> None:
    assert help_text(None, compact=False, route="list", auto_follow=False) == (
        "Help\n↑/↓ select · h close help · q quit · r resume output"
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


def test_events_text_empty() -> None:
    assert events_text(()) == "Events\nNo events yet."


def test_events_text_non_empty() -> None:
    assert events_text(("e1", "e2")) == "Events\ne1\ne2"


def test_run_row_active_with_retry_uses_bold_red_status_cell() -> None:
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
    assert isinstance(status_cell, Text)
    assert status_cell.plain == "running 2/3"
    assert status_cell.style == "bold red"
    assert progress == "█" * 5 + "░" * 5 + " 55%"
    assert elapsed == "02:05"


def test_run_row_active_without_retry_uses_status_style() -> None:
    run = active_run(
        status=ExecutionRunStatus.RUNNING,
        attempt=1,
        max_attempts=3,
    )
    _, _, status_cell, _, _ = run_row(run)

    assert status_cell.plain == "running"
    assert status_cell.style == "cyan"


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
    assert status_cell.style == "green"
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


def test_confirmation_text_force_stop() -> None:
    assert confirmation_text("force", "run-9", 1) == "Force stop run-9? [y] confirm [n] cancel"


def test_confirmation_text_quit_singular() -> None:
    assert confirmation_text("quit", None, 1) == (
        "Stop scheduling and gracefully stop 1 active run? [y] confirm [n] cancel"
    )


def test_confirmation_text_quit_plural() -> None:
    assert confirmation_text("quit", None, 2) == (
        "Stop scheduling and gracefully stop 2 active runs? [y] confirm [n] cancel"
    )
