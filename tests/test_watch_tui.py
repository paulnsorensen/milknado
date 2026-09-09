from __future__ import annotations

import inspect
from dataclasses import replace
from pathlib import Path
from typing import cast, get_type_hints

import pytest
from rich.text import Text
from textual.widgets import Static

import milknado.app.watch_tui as watch_tui
from milknado.app.run import (
    ActiveRunSnapshot,
    ExecutionController,
    ExecutionRunStatus,
    ExecutionSnapshot,
    RunActionAvailability,
)
from milknado.app.run_source import ExecutionSnapshotSource
from milknado.app.run_tui import ExecutionApp
from milknado.app.run_view_app import ExecutionSnapshotApp
from milknado.app.watch import WatchSnapshotSource


class FakeSource:
    def __init__(self, snapshot: ExecutionSnapshot) -> None:
        self.current: ExecutionSnapshot = snapshot

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
        actions = {active.binding.action for active in app.screen.active_bindings.values()}
        assert actions.isdisjoint({"focus_guidance", "cancel", "force"})


@pytest.mark.asyncio
@pytest.mark.parametrize("quit_key", ["q", "ctrl+c", "ctrl+q"])
async def test_watch_quit_exits_without_controlling_the_observed_run(quit_key: str) -> None:
    app = watch_tui.WatchApp(FakeSource(snapshot()))

    async with app.run_test() as pilot:
        await pilot.press(quit_key)
        assert not app.is_running


def test_watch_tui_entry_builds_source_and_discards_app_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    expected = object()

    class FakeApp:
        def __init__(self, source: WatchSnapshotSource) -> None:
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
    controller = watch_tui._WatchController  # pyright: ignore[reportPrivateUsage] -- exercising the read-only adapter's Protocol shape directly
    assert list(inspect.signature(controller.subscribe).parameters) == ["listener"]


@pytest.mark.asyncio
async def test_watch_help_overlay_is_visible_and_excludes_operator_actions() -> None:
    active = ActiveRunSnapshot(
        run_id="watch-active",
        node_id=1,
        description="Observed run",
        status=ExecutionRunStatus.RUNNING,
        progress="Working",
        stop_requested=False,
        actions=RunActionAvailability("Read-only", "Read-only", "Read-only"),
        output=(),
        pending_guidance=None,
        elapsed_seconds=0.0,
        progress_pct=None,
        eta_seconds=None,
        attempt=1,
        max_attempts=1,
        stalled=False,
    )
    app = watch_tui.WatchApp(FakeSource(replace(snapshot(), active_runs=(active,))))

    async with app.run_test(size=(40, 15)) as pilot:
        await pilot.pause()
        await pilot.press("f1")

        overlay = app.query_one("#help-overlay", Static)
        assert overlay.has_class("visible")
        assert "Help" in app.export_screenshot().replace("&#160;", " ")
        help_text = cast(Text, overlay.render()).plain
        assert all(label not in help_text for label in ("g queue guidance", "c cancel", "f force"))
