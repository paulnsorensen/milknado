from __future__ import annotations

import argparse
from collections.abc import Callable
from datetime import datetime, tzinfo

import textual.widgets._header as header

from milknado.app.run import (
    ActiveRunSnapshot,
    ExecutionRunStatus,
    ExecutionSnapshot,
    RunActionAvailability,
)
from milknado.app.run_source import NodeSnapshotRequest
from milknado.app.run_tui import ExecutionApp
from milknado.app.watch_tui import WatchApp
from milknado.domains.graph import NodeDetailResponse


class FixedDatetime(datetime):
    @classmethod
    def now(cls, tz: tzinfo | None = None) -> FixedDatetime:
        return cls(2026, 1, 1, 12, 0, 0, tzinfo=tz)


header.datetime = FixedDatetime


class FixtureSource:
    def __init__(self, current: ExecutionSnapshot) -> None:
        self.current = current

    def snapshot(self) -> ExecutionSnapshot:
        return self.current

    def attached_watch_source(self) -> FixtureSource:
        return self

    def subscribe(self, listener: Callable[[ExecutionSnapshot], None]) -> Callable[[], None]:
        listener(self.current)
        return lambda: None

    def node_snapshot(self, request: NodeSnapshotRequest) -> NodeDetailResponse:
        return NodeDetailResponse(request.node_id, request.request_generation, None)


def fixture(state: str) -> ExecutionSnapshot:
    retry = state == "retry-error"
    output = (
        ("worker exited with code 1", "retry scheduled")
        if retry
        else ("worker start outcome unknown",)
    )
    run = ActiveRunSnapshot(
        run_id="run-1",
        node_id=1,
        description="Retry failed node after worker exit" if retry else "Unknown-start held node",
        status=ExecutionRunStatus.RUNNING,
        progress=(
            "Retry scheduled after worker error" if retry else "Start outcome unknown; hold retry"
        ),
        stop_requested=False,
        actions=RunActionAvailability(),
        output=output,
        pending_guidance=(),
        elapsed_seconds=12.0,
        progress_pct=50.0 if retry else None,
        eta_seconds=8.0 if retry else None,
        attempt=2 if retry else 1,
        max_attempts=3,
        stalled=not retry,
    )
    return ExecutionSnapshot(
        goal="Bounded node attempts",
        active_runs=(run,),
        terminal_runs=(),
        completed=1,
        failed=0,
        stopped=0,
        available=2,
        event_lines=output,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("surface", choices=("run", "watch"))
    parser.add_argument("state", choices=("retry-error", "unknown-start"))
    args = parser.parse_args()
    source = FixtureSource(fixture(args.state))
    app = ExecutionApp(source) if args.surface == "run" else WatchApp(source, poll_interval=60)
    app.run()


if __name__ == "__main__":
    main()
