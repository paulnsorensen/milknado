from __future__ import annotations

import inspect
from dataclasses import replace
from typing import get_type_hints
from unittest.mock import patch

import pytest

import milknado.app.watch_tui as watch_tui
from milknado.app.run import ExecutionController, ExecutionSnapshot
from milknado.app.run_source import ExecutionSnapshotSource
from milknado.app.run_tui import ExecutionApp
from milknado.app.run_view_app import ExecutionSnapshotApp


class FakeSource:
    def __init__(self, snapshot: ExecutionSnapshot) -> None:
        self.current = snapshot

    def snapshot(self) -> ExecutionSnapshot:
        return self.current


def snapshot(goal: str = "Initial goal") -> ExecutionSnapshot:
    return ExecutionSnapshot(
        goal=goal,
        active_runs=(),
        terminal_runs=(),
        completed=0,
        failed=0,
        stopped=0,
        available=0,
        event_lines=(),
    )


@pytest.mark.asyncio
async def test_watch_app_refreshes_from_source_without_control_bindings() -> None:
    source = FakeSource(snapshot())
    app = watch_tui.WatchApp(source, poll_interval=60.0)

    async with app.run_test(size=(120, 36)):
        assert app.title == "Initial goal"
        source.current = replace(source.current, goal="Refreshed goal", available=2)
        app.poll()
        assert app.title == "Refreshed goal"
        assert app.sub_title.endswith("2 available")
        assert [tuple(binding[:3]) for binding in app.BINDINGS] == [
            ("up", "previous_run", "Previous run"),
            ("down", "next_run", "Next run"),
            ("enter", "open_detail", "Open selected run"),
            ("escape", "back", "Back"),
            ("r", "resume_output", "Resume output"),
            ("h", "help", "Help"),
            ("q", "quit_all", "Quit"),
        ]
        assert {binding[1] for binding in app.BINDINGS}.isdisjoint(
            {"focus_guidance", "cancel", "force"}
        )


def test_watch_quit_exits_without_controlling_the_observed_run() -> None:
    app = watch_tui.WatchApp(FakeSource(snapshot()))

    with patch.object(app, "exit") as exit_app:
        app.action_quit_all()

    exit_app.assert_called_once_with()


def test_watch_tui_entry_builds_source_and_discards_app_result(monkeypatch, tmp_path) -> None:
    expected = object()

    class FakeApp:
        def __init__(self, source: object) -> None:
            assert source.project_root == tmp_path
            assert source.db_path == tmp_path / "milknado.db"

        def run(self) -> object:
            return expected

    monkeypatch.setattr(watch_tui, "WatchApp", FakeApp)

    assert watch_tui.run_watch_tui(tmp_path, tmp_path / "milknado.db") is None


def test_snapshot_view_has_no_execution_controller_contract() -> None:
    assert issubclass(watch_tui.WatchApp, ExecutionSnapshotApp)
    assert not issubclass(watch_tui.WatchApp, ExecutionApp)
    assert get_type_hints(ExecutionSnapshotApp.__init__)["source"] is ExecutionSnapshotSource
    assert get_type_hints(ExecutionApp.__init__)["controller"] is ExecutionController
    assert list(inspect.signature(watch_tui._WatchController.subscribe).parameters) == ["listener"]
