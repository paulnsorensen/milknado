"""Snapshot tests for TUI builders — US-103 from issue #12.

Tests call the display module's pure functions directly and assert on the
rendered output via ``Console.render_lines``. No ``Live`` terminal is spawned.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Generator
from pathlib import Path

import pytest
from rich.console import Console

from milknado.domains.common.protocols import ProgressEvent
from milknado.domains.execution.run_loop import display
from milknado.domains.execution.run_loop.input import InputController
from milknado.domains.graph import MikadoGraph


@pytest.fixture()
def graph(tmp_path: Path) -> Generator[MikadoGraph, None, None]:
    g = MikadoGraph(tmp_path / "display.db")
    yield g
    g.close()


def _plain(renderable: object) -> str:
    console = Console(width=200, record=True, color_system=None)
    console.print(renderable)
    return console.export_text()


class TestRenderProgressBar:
    def test_normal_spinner_frame(self) -> None:
        out = display.render_progress_bar(
            frame="◜", elapsed=10.0, pct=None, stall_threshold=300,
        )
        assert "◜" in out
        assert "⚠" not in out
        assert "cyan" in out

    def test_stalled_glyph(self) -> None:
        out = display.render_progress_bar(
            frame="◜", elapsed=301.0, pct=None, stall_threshold=300,
        )
        assert "⚠" in out
        assert "yellow" in out

    def test_completed_bar_at_100_pct(self) -> None:
        out = display.render_progress_bar(
            frame="◜", elapsed=5.0, pct=100.0, stall_threshold=300,
        )
        assert "█" * 10 in out
        assert "100%" in out
        assert "⚠" not in out

    def test_partial_progress_bar(self) -> None:
        out = display.render_progress_bar(
            frame="◜", elapsed=5.0, pct=30.0, stall_threshold=300,
        )
        assert "███" in out
        assert "░" * 7 in out
        assert "30%" in out


class TestBuildTitle:
    def test_emits_aggregate_counts(self) -> None:
        title = display.build_title(
            active_count=2, done=3, failed=1, blocked=0,
        )
        assert "2 active" in title
        assert "3 done" in title
        assert "1 failed" in title
        assert "0 blocked" in title


class TestBuildLogPanel:
    def test_renders_log_lines(self) -> None:
        logs = deque(["[12:00:00] → node 1", "[12:00:05] ✓ node 1"], maxlen=30)
        panel = display.build_log_panel(logs)
        text = _plain(panel)
        assert "→ node 1" in text
        assert "✓ node 1" in text

    def test_empty_shows_placeholder(self) -> None:
        panel = display.build_log_panel(deque(maxlen=30))
        text = _plain(panel)
        assert "No events yet" in text

    def test_deque_truncation_at_maxlen(self) -> None:
        logs: deque[str] = deque(maxlen=30)
        for i in range(50):
            logs.append(f"line {i}")
        panel = display.build_log_panel(logs)
        text = _plain(panel)
        assert "line 49" in text
        assert "line 20" in text
        assert "line 19" not in text


class TestBuildWorkerTable:
    def test_columns_present_when_worker_active(self, graph: MikadoGraph) -> None:
        node = graph.add_node("apply fix")
        graph.mark_running(node.id, "/tmp/wt", "branch-a", "run-abc")
        graph.set_file_ownership(node.id, ["src/a.py", "src/b.py"])
        table = display.build_worker_table(
            tick=1,
            now=100.0,
            active={"run-abc": node.id},
            dispatched_at={"run-abc": 40.0},
            attempts={},
            progress_by_run={},
            completion_durations=[],
            graph=graph,
            stall_threshold=300,
            max_retries=2,
        )
        text = _plain(table)
        assert "apply fix" in text
        assert "src/a.py" in text
        assert "01:00" in text  # 60s elapsed

    def test_shows_attempts_column_when_retrying(self, graph: MikadoGraph) -> None:
        node = graph.add_node("flaky node")
        graph.mark_running(node.id, "/tmp/wt", "branch-b", "run-2")
        graph.set_file_ownership(node.id, ["f.py"])
        table = display.build_worker_table(
            tick=0,
            now=10.0,
            active={"run-2": node.id},
            dispatched_at={"run-2": 5.0},
            attempts={node.id: 1},
            progress_by_run={},
            completion_durations=[],
            graph=graph,
            stall_threshold=300,
            max_retries=2,
        )
        text = _plain(table)
        assert "2/3" in text

    def test_eta_shown_after_three_completions(self, graph: MikadoGraph) -> None:
        node = graph.add_node("n")
        graph.mark_running(node.id, "/tmp/wt", "branch-c", "run-3")
        graph.set_file_ownership(node.id, [])
        table = display.build_worker_table(
            tick=0,
            now=20.0,
            active={"run-3": node.id},
            dispatched_at={"run-3": 0.0},
            attempts={},
            progress_by_run={},
            completion_durations=[60.0, 60.0, 60.0],
            graph=graph,
            stall_threshold=300,
            max_retries=2,
        )
        text = _plain(table)
        assert "~00:40" in text  # avg 60s - elapsed 20s = 40s remaining

    def test_progress_bar_takes_precedence_over_spinner(
        self, graph: MikadoGraph,
    ) -> None:
        node = graph.add_node("reporting node")
        graph.mark_running(node.id, "/tmp/wt", "branch-d", "run-4")
        graph.set_file_ownership(node.id, [])
        table = display.build_worker_table(
            tick=0,
            now=5.0,
            active={"run-4": node.id},
            dispatched_at={"run-4": 0.0},
            attempts={},
            progress_by_run={
                "run-4": ProgressEvent(run_id="run-4", work=5, total=10),
            },
            completion_durations=[],
            graph=graph,
            stall_threshold=300,
            max_retries=2,
        )
        text = _plain(table)
        # "50%" is truncated by the 12-char progress column, but the bar
        # pattern still proves the progress bar beat the spinner.
        assert "█████░░░░░" in text
        assert "◜" not in text


class TestRenderOverlay:
    def test_renders_selected_worker_metadata_and_stdout(
        self, graph: MikadoGraph,
    ) -> None:
        node = graph.add_node("wire the thing")
        graph.mark_running(node.id, "/tmp/wt", "feat/milknado-1", "run-xyz")
        panel = display.render_overlay(
            run_id="run-xyz",
            active={"run-xyz": node.id},
            graph=graph,
            agent="claude --model sonnet",
            stdout_lines=[f"line {i}" for i in range(120)],
        )
        text = _plain(panel)
        assert "feat/milknado-1" in text
        assert "claude --model sonnet" in text
        assert "wire the thing" in text
        assert "line 119" in text
        assert "line 20" in text  # last 100 of 120 keeps [20..119]
        assert "line 19" not in text

    def test_unknown_run_shows_placeholder(self, graph: MikadoGraph) -> None:
        panel = display.render_overlay(
            run_id="ghost",
            active={},
            graph=graph,
            agent="x",
            stdout_lines=[],
        )
        text = _plain(panel)
        assert "worker not found" in text

    def test_empty_stdout_shows_placeholder(self, graph: MikadoGraph) -> None:
        node = graph.add_node("quiet node")
        graph.mark_running(node.id, "/tmp/wt", "b", "run-q")
        panel = display.render_overlay(
            run_id="run-q",
            active={"run-q": node.id},
            graph=graph,
            agent="x",
            stdout_lines=[],
        )
        text = _plain(panel)
        assert "no output yet" in text


class TestInputController:
    def test_l_digits_enter_opens_overlay(self) -> None:
        opened: list[int] = []
        cleared: list[bool] = []
        ctrl = InputController(
            on_overlay=opened.append,
            on_clear=lambda: cleared.append(True),
        )
        for ch in "L42\r":
            ctrl.inject(ch)
        assert opened == [42]
        assert cleared == []

    def test_esc_clears_overlay(self) -> None:
        opened: list[int] = []
        cleared: list[bool] = []
        ctrl = InputController(
            on_overlay=opened.append,
            on_clear=lambda: cleared.append(True),
        )
        ctrl.inject("\x1b")
        assert cleared == [True]

    def test_enter_without_digits_is_noop(self) -> None:
        opened: list[int] = []
        ctrl = InputController(
            on_overlay=opened.append,
            on_clear=lambda: None,
        )
        ctrl.inject("L")
        ctrl.inject("\r")
        assert opened == []

    def test_non_digit_keys_ignored_inside_digit_phase(self) -> None:
        opened: list[int] = []
        ctrl = InputController(
            on_overlay=opened.append,
            on_clear=lambda: None,
        )
        for ch in "L1x2\r":
            ctrl.inject(ch)
        # 'x' is ignored; digits still accumulate.
        assert opened == [12]


class TestHelpers:
    def test_elapsed_str_formats_mm_ss(self) -> None:
        assert display.elapsed_str(0.0) == "00:00"
        assert display.elapsed_str(65.7) == "01:05"
        assert display.elapsed_str(3599.0) == "59:59"

    def test_eta_str_unknown_when_no_avg(self) -> None:
        assert display.eta_str(None, elapsed=10.0) == "~?"

    def test_eta_str_clamps_at_zero(self) -> None:
        assert display.eta_str(5.0, elapsed=100.0) == "~00:00"

    def test_files_cell_truncates_over_60(self) -> None:
        files = [f"f{i}.py" for i in range(20)]
        out = display.files_cell(files)
        assert len(out) == 60
        assert out.endswith("…")

    def test_files_cell_short_stays_short(self) -> None:
        assert display.files_cell(["a.py", "b.py"]) == "a.py, b.py"

    def test_is_stalled_boundary(self) -> None:
        assert display.is_stalled(300.0, 300) is True
        assert display.is_stalled(299.9, 300) is False
