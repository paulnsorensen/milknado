"""Read-only projection of durable graph and run state for observers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import msgspec

from milknado.app.run import (
    ActiveRunSnapshot,
    ExecutionRunStatus,
    ExecutionSnapshot,
    RunActionAvailability,
    TerminalRunSnapshot,
)
from milknado.domains.dispatch import tail, tail_latest_iteration_log
from milknado.domains.execution import get_execution_overview
from milknado.domains.graph import MikadoGraph

_OBSERVER_ACTIONS = RunActionAvailability(
    cancel_reason="Observer mode is read-only.",
    guidance_reason="Observer mode is read-only.",
    force_stop_reason="Observer mode is read-only.",
)


class DurableRun(msgspec.Struct, frozen=True):
    """Validated runs-table record used by the observer projection."""

    run_id: str
    node_id: int
    status: Literal["running", "done", "failed"]
    pid: int | None
    log_path: str
    started_at: str
    ended_at: str | None
    timed_out: bool
    exit_code: int | None
    error: str | None
    timeout_seconds: int | None
    detail: str | None
    rebased: bool | None


@dataclass(frozen=True, slots=True)
class WatchSnapshotSource:
    """Build bounded presentation snapshots without opening a writer connection."""

    project_root: Path
    db_path: Path
    limit: int = 50

    def snapshot(self) -> ExecutionSnapshot:
        graph = MikadoGraph.open_snapshot(self.db_path)
        try:
            runs = tuple(
                msgspec.convert(row, type=DurableRun, strict=True)
                for row in graph.recent_runs(self.limit)
            )
            goal, descriptions, available = get_execution_overview(
                graph,
                [run.node_id for run in runs],
            )
        finally:
            graph.close()
        active = tuple(
            self._active_snapshot(run, descriptions.get(run.node_id, str(run.node_id)))
            for run in runs
            if run.status == "running"
        )
        terminal = tuple(
            self._terminal_snapshot(run, descriptions.get(run.node_id, str(run.node_id)))
            for run in reversed(runs)
            if run.status != "running"
        )
        return ExecutionSnapshot(
            goal=goal or str(self.project_root),
            active_runs=active,
            terminal_runs=terminal,
            completed=sum(run.status == "done" for run in runs),
            failed=sum(run.status == "failed" for run in runs),
            stopped=0,
            available=available,
            event_lines=tuple(f"{run.run_id} · {run.status}" for run in reversed(runs[:20])),
        )

    def _active_snapshot(self, run: DurableRun, description: str) -> ActiveRunSnapshot:
        return ActiveRunSnapshot(
            run_id=run.run_id,
            node_id=run.node_id,
            description=description,
            status=ExecutionRunStatus.RUNNING,
            progress=None,
            stop_requested=False,
            actions=_OBSERVER_ACTIONS,
            output=self._output(run),
            pending_guidance=(),
            elapsed_seconds=self._duration(run.started_at, None),
            progress_pct=None,
            eta_seconds=None,
            attempt=1,
            max_attempts=1,
            stalled=False,
        )

    def _terminal_snapshot(self, run: DurableRun, description: str) -> TerminalRunSnapshot:
        status = (
            ExecutionRunStatus.COMPLETED if run.status == "done" else ExecutionRunStatus.FAILED
        )
        return TerminalRunSnapshot(
            run_id=run.run_id,
            node_id=run.node_id,
            description=description,
            status=status,
            output=self._output(run),
            pending_guidance=(),
            duration_seconds=self._duration(run.started_at, run.ended_at),
        )

    def _output(self, run: DurableRun) -> tuple[str, ...]:
        candidate = Path(run.log_path).resolve()
        if not candidate.is_relative_to(self.project_root.resolve()):
            return ()
        text = tail_latest_iteration_log(candidate) if candidate.is_dir() else tail(candidate)
        return tuple(text.splitlines())

    @staticmethod
    def _duration(started_at: str, ended_at: str | None) -> float:
        started = datetime.fromisoformat(started_at)
        ended = datetime.fromisoformat(ended_at) if ended_at else datetime.now(UTC)
        return max(0.0, (ended - started).total_seconds())
