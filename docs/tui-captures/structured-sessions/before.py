from __future__ import annotations

import json

from fixture import ASSISTANT_TEXT, BRIEF, FOLLOW_UP_TEXT, TOOL_TEXT, USER_TEXT, Source, run_demo

from milknado.app.run import (
    ActiveRunSnapshot,
    ExecutionRunStatus,
    ExecutionSnapshot,
    RunActionAvailability,
    TerminalRunSnapshot,
)


def baseline_snapshot() -> ExecutionSnapshot:
    output = tuple(
        json.dumps({"type": "message_end", "message": {"role": role, "content": text}})
        for role, text in (
            ("user", USER_TEXT),
            ("assistant", ASSISTANT_TEXT),
            ("toolResult", TOOL_TEXT),
            ("user", "Keep watch read-only."),
            ("assistant", FOLLOW_UP_TEXT),
        )
    )
    active = ActiveRunSnapshot(
        run_id="demo-active",
        node_id=12,
        description=BRIEF,
        status=ExecutionRunStatus.RUNNING,
        progress="Implementing session views",
        stop_requested=False,
        actions=RunActionAvailability(),
        output=output,
        pending_guidance=(),
        elapsed_seconds=125,
        progress_pct=45,
        eta_seconds=150,
        attempt=1,
        max_attempts=3,
        stalled=False,
    )
    failed = TerminalRunSnapshot(
        run_id="demo-failed",
        node_id=13,
        description="Protect completion gates",
        status=ExecutionRunStatus.FAILED,
        output=("Quality gate rejected an incomplete change.",),
        pending_guidance=(),
        duration_seconds=83,
    )
    return ExecutionSnapshot(
        goal="Milknado TUI dogfood",
        active_runs=(active,),
        terminal_runs=(failed,),
        completed=0,
        failed=1,
        stopped=0,
        available=2,
        event_lines=("Node 12 started", "Node 13 failed its quality gate"),
    )


if __name__ == "__main__":
    run_demo(Source(baseline_snapshot()))
