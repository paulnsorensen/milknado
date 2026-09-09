"""Render controlled snapshots; this fixture never launches or controls a worker."""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import replace
from threading import Event
from typing import cast
from unittest.mock import patch

from rich.text import Text

from milknado.app.run import (
    ActiveRunSnapshot,
    ExecutionController,
    ExecutionRunStatus,
    ExecutionSnapshot,
    RunActionAvailability,
    TerminalRunSnapshot,
)
from milknado.app.run_tui import ExecutionApp
from milknado.app.watch_tui import WatchApp


class Source:
    def __init__(self, mode: str, state: str) -> None:
        self.stopped = Event()
        actions = (
            RunActionAvailability()
            if mode == "run"
            else RunActionAvailability(*("Observer mode is read-only.",) * 3)
        )
        active = ActiveRunSnapshot(
            "demo-active",
            12,
            "Repair compact keyboard navigation",
            ExecutionRunStatus.RUNNING,
            "Implementing navigation",
            False,
            actions,
            ("Inspecting the dashboard", "Checking keyboard focus", "Review pending"),
            (),
            125,
            45,
            150,
            1,
            3,
            False,
        )
        failed = TerminalRunSnapshot(
            "demo-failed",
            13,
            "Keep intervention errors visible",
            ExecutionRunStatus.FAILED,
            ("Quality gate failed: narrow layout",),
            (),
            83,
        )
        self.current = ExecutionSnapshot(
            "Milknado TUI dogfood",
            (active,),
            (failed,),
            0,
            1,
            0,
            2,
            ("Node 12 started", "Node 13 failed its quality gate"),
        )
        if state == "error":
            self.current = replace(self.current, listener_errors=("Snapshot source unavailable",))
        elif state == "empty":
            self.current = replace(
                self.current, active_runs=(), terminal_runs=(), failed=0, event_lines=()
            )

    def snapshot(self) -> ExecutionSnapshot:
        return self.current

    @staticmethod
    def subscribe(_listener: Callable[[ExecutionSnapshot], None]) -> Callable[[], None]:
        return lambda: None

    def run(self, **_kwargs: object) -> None:
        if not self.stopped.wait(timeout=120):
            raise TimeoutError("Demo did not receive a quit request")

    def stop_scheduling(self) -> None:
        self.stopped.set()


if __name__ == "__main__":
    mode, state = sys.argv[1:3]
    source = Source(mode, state)
    with patch("textual.widgets._header.HeaderClock.render", return_value=Text("12:00:00")):
        app = (
            ExecutionApp(cast(ExecutionController, cast(object, source)), feature_branch="demo")
            if mode == "run"
            else WatchApp(source)
        )
        _ = app.run()
