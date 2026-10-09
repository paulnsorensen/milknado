"""Run the real Textual views with fixed, offline presentation data."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from dataclasses import replace
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
from milknado.app.run_source import NodeSnapshotRequest
from milknado.app.run_tui import ExecutionApp
from milknado.app.watch_tui import WatchApp
from milknado.domains.graph import NodeDetailResponse

CAPTURE_LIMITS = (
    "OFFLINE FIXTURE: fixed goal, run, status, output, elapsed time, and clock. "
    "The real Textual view renders, but no worker, credential, durable graph, "
    "or control authority is exercised."
)


def _active_run_snapshot() -> ActiveRunSnapshot:
    unavailable = "Offline capture has no control authority."
    return ActiveRunSnapshot(
        run_id="goal-4-claude",
        node_id=4,
        description="Recover Claude execution status",
        status=ExecutionRunStatus.RUNNING,
        progress="Pending recovery; provider state unknown",
        stop_requested=False,
        actions=RunActionAvailability(unavailable, unavailable, unavailable),
        output=(
            "Coordinator status: pending recovery",
            "Provider: Claude",
            "Worker status: unknown (fixture only)",
        ),
        pending_guidance=(),
        elapsed_seconds=125,
        progress_pct=45,
        eta_seconds=150,
        attempt=1,
        max_attempts=3,
        stalled=False,
    )


def _failed_run_snapshot() -> TerminalRunSnapshot:
    return TerminalRunSnapshot(
        run_id="goal-5-gate",
        node_id=5,
        description="Keep the quality gate visible",
        status=ExecutionRunStatus.FAILED,
        output=("The quality gate did not pass.",),
        pending_guidance=(),
        duration_seconds=83,
    )


def fixture_snapshot(state: str) -> ExecutionSnapshot:
    snapshot = ExecutionSnapshot(
        goal="Milknado PR 520 review fixture",
        active_runs=(_active_run_snapshot(),),
        terminal_runs=(_failed_run_snapshot(),),
        completed=0,
        failed=1,
        stopped=0,
        available=2,
        event_lines=(
            "Goal 4: Claude pending recovery; worker status unknown",
            "Goal 5: quality gate failed",
        ),
    )
    if state == "error":
        return replace(snapshot, listener_errors=("Snapshot source unavailable",))
    if state == "empty":
        return replace(snapshot, active_runs=(), terminal_runs=(), failed=0, event_lines=())
    return snapshot


class Source:
    def __init__(self, current: ExecutionSnapshot) -> None:
        self.current = current

    def snapshot(self) -> ExecutionSnapshot:
        return self.current

    def attached_watch_source(self) -> Source:
        return self

    @staticmethod
    def coordinator_status() -> str:
        return "Goal 4 · claude · pending · recovery: unknown"

    @staticmethod
    def node_snapshot(request: NodeSnapshotRequest) -> NodeDetailResponse:
        return NodeDetailResponse(request.node_id, request.request_generation, None)

    @staticmethod
    def subscribe(
        listener: Callable[[ExecutionSnapshot], None],
    ) -> Callable[[], None]:
        del listener
        return lambda: None

    @staticmethod
    def force_stop_all(timeout: float = 8.0) -> bool:
        del timeout
        return True

    @staticmethod
    def stop_scheduling() -> None:
        return None


def make_app(view: str, state: str, *, attached: bool = False) -> WatchApp | ExecutionApp:
    source = Source(fixture_snapshot(state))
    if view == "watch":
        return WatchApp(source, read_only=not attached)
    if attached:
        raise ValueError("--attached requires --view watch")
    return ExecutionApp(cast(ExecutionController, cast(object, source)), feature_branch=None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view", choices=("watch", "run"), required=True)
    parser.add_argument("--state", choices=("normal", "error", "empty"), default="normal")
    parser.add_argument("--attached", action="store_true", help="show the attached watch view")
    args = parser.parse_args()
    if args.attached and args.view != "watch":
        parser.error("--attached requires --view watch")
    print(CAPTURE_LIMITS, file=sys.stderr, flush=True)
    with patch("textual.widgets._header.HeaderClock.render", return_value=Text("12:00:00")):
        _ = make_app(args.view, args.state, attached=args.attached).run()


if __name__ == "__main__":
    main()
