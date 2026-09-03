from __future__ import annotations

from dataclasses import replace
from unittest.mock import patch

import pytest

from milknado.app.run import ExecutionSnapshot
from milknado.app.watch_tui import WatchApp, run_watch_tui


class FakeSource:
    def __init__(self, snapshot: ExecutionSnapshot) -> None:
        self.current = snapshot
        self.calls = 0

    def snapshot(self) -> ExecutionSnapshot:
        self.calls += 1
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
    app = WatchApp(source, poll_interval=60.0)

    async with app.run_test(size=(120, 36)):
        assert app.title == "Initial goal"
        source.current = replace(source.current, goal="Refreshed goal", available=2)
        app.poll()
        assert app.title == "Refreshed goal"
        assert app.sub_title.endswith("2 available")
        assert {binding[0] for binding in app.BINDINGS} == {
            "up",
            "down",
            "enter",
            "escape",
            "r",
            "h",
            "q",
        }


def test_watch_quit_exits_without_controlling_the_observed_run() -> None:
    app = WatchApp(FakeSource(snapshot()))

    with patch.object(app, "exit") as exit_app:
        app.action_quit_all()

    exit_app.assert_called_once_with()


def test_watch_tui_entry_builds_source_and_runs_app(monkeypatch, tmp_path) -> None:
    expected = object()

    class FakeApp:
        def __init__(self, source: object) -> None:
            assert source.project_root == tmp_path
            assert source.db_path == tmp_path / "milknado.db"

        def run(self) -> object:
            return expected

    import milknado.app.watch_tui as watch_tui

    monkeypatch.setattr(watch_tui, "WatchApp", FakeApp)

    assert run_watch_tui(tmp_path, tmp_path / "milknado.db") is expected
