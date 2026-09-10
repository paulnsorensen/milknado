from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, cast
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

if TYPE_CHECKING:
    from milknado.domains.common import SessionInput

BRIEF = (
    "Improve structured sessions\n\n"
    "Decode OMP, Claude, and Codex messages. Preserve delivery receipts and completion gates.\n\n"
    "Keep the full task brief available in Details."
)
USER_TEXT = "Improve the session view without bypassing quality gates."
ASSISTANT_TEXT = "I found the raw-output path. I will decode messages before the UI renders them."
TOOL_TEXT = "Read session.py\nThe current title exposes raw worker output."
FOLLOW_UP_TEXT = "Watch stays read-only. Session, Changes, and Details share one layout."


def seed_repository(root: Path) -> str:
    environment = os.environ | {
        "GIT_AUTHOR_DATE": "2026-09-09T12:00:00+00:00",
        "GIT_COMMITTER_DATE": "2026-09-09T12:00:00+00:00",
    }

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=root, env=environment, check=True, capture_output=True, text=True
        ).stdout.strip()

    git("init", "-q")
    git("config", "user.name", "Demo")
    git("config", "user.email", "demo@example.invalid")
    (root / "session.py").write_text("# Render raw output.\n", encoding="utf-8")
    git("add", "session.py")
    git("commit", "-qm", "Seed the session demo")
    base = git("rev-parse", "HEAD")
    (root / "session.py").write_text("# Render structured messages.\n", encoding="utf-8")
    (root / "policy.txt").write_text(
        "Preserve permission policy and completion gates.\n", encoding="utf-8"
    )
    return base


def initial_snapshot(root: Path, state: str) -> ExecutionSnapshot:
    from milknado.domains.common import SessionContext, SessionEvent, SessionView

    base = seed_repository(root)
    events = (
        SessionEvent(kind="user", text=USER_TEXT, event_id="prompt", state="delivered"),
        SessionEvent(
            kind="assistant", text=ASSISTANT_TEXT, event_id="message-1", state="complete"
        ),
        SessionEvent(kind="tool", text=TOOL_TEXT, event_id="tool-1", state="completed"),
        SessionEvent(
            kind="user",
            text="Keep watch read-only.",
            event_id="input-1",
            action="steer",
            state="delivered",
        ),
        SessionEvent(
            kind="assistant", text=FOLLOW_UP_TEXT, event_id="message-2", state="streaming"
        ),
    )
    if state == "error":
        events += (
            SessionEvent(
                kind="error",
                text="Worker connection closed before its input receipt.",
                state="error",
            ),
        )
    session = SessionView(
        context=SessionContext(family="omp", cwd=str(root), base_oid=base),
        events=events,
        actions=("steer", "follow_up", "interrupt", "approve", "deny"),
        active=True,
    )
    active = ActiveRunSnapshot(
        run_id="demo-active",
        node_id=12,
        description=BRIEF,
        status=ExecutionRunStatus.RUNNING,
        progress="Implementing session views",
        stop_requested=False,
        actions=RunActionAvailability(),
        output=("Raw protocol frames stay in the diagnostic log.",),
        pending_guidance=(),
        elapsed_seconds=125,
        progress_pct=45,
        eta_seconds=150,
        attempt=1,
        max_attempts=3,
        stalled=False,
        session=session,
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
    snapshot = ExecutionSnapshot(
        goal="Milknado TUI dogfood",
        active_runs=(active,),
        terminal_runs=(failed,),
        completed=0,
        failed=1,
        stopped=0,
        available=2,
        event_lines=("Node 12 started", "Node 13 failed its quality gate"),
    )
    if state == "empty":
        return replace(snapshot, active_runs=(), terminal_runs=(), failed=0, event_lines=())
    return snapshot


class Source:
    def __init__(self, current: ExecutionSnapshot) -> None:
        self.current = current
        self.listeners: list[Callable[[ExecutionSnapshot], None]] = []

    def snapshot(self) -> ExecutionSnapshot:
        return self.current

    def subscribe(self, listener: Callable[[ExecutionSnapshot], None]) -> Callable[[], None]:
        self.listeners.append(listener)
        return lambda: self.listeners.remove(listener)

    def session_input(self, run_id: str, command: SessionInput) -> bool:
        from milknado.domains.common import SessionEvent

        if not self.current.active_runs or self.current.active_runs[0].run_id != run_id:
            return False
        run = self.current.active_runs[0]
        event = SessionEvent(
            kind="user",
            text=command.text,
            event_id="demo-input",
            action=command.action,
            state="delivered",
        )
        self.current = replace(
            self.current,
            active_runs=(
                replace(run, session=replace(run.session, events=(*run.session.events, event))),
            ),
        )
        self._publish()
        return True

    def stop_scheduling(self) -> None:
        self.current = replace(self.current, active_runs=(), stopped=1)
        self._publish()

    def cancel(self, run_id: str) -> None:
        if any(run.run_id == run_id for run in self.current.active_runs):
            self.stop_scheduling()

    def force_stop(self, run_id: str, _timeout: float = 10) -> bool:
        if not any(run.run_id == run_id for run in self.current.active_runs):
            return False
        self.stop_scheduling()
        return True

    def _publish(self) -> None:
        for listener in self.listeners:
            listener(self.current)


def run_demo(source: Source, *, read_only: bool = False) -> None:
    with patch("textual.widgets._header.HeaderClock.render", return_value=Text("12:00:00")):
        app = WatchApp(source) if read_only else ExecutionApp(cast(ExecutionController, source))
        _ = app.run()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "main"
    repository = Path("/tmp/milknado-session-demo")
    repository.mkdir(mode=0o700)
    try:
        run_demo(Source(initial_snapshot(repository, mode)), read_only=mode == "watch")
    finally:
        shutil.rmtree(repository)
