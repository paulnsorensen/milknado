"""Read-only projection of durable graph and run state for observers."""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO

from milknado.app.run import (
    ActiveRunSnapshot,
    ExecutionRunStatus,
    ExecutionSnapshot,
    RunActionAvailability,
    TerminalRunSnapshot,
)
from milknado.domains.graph import DurableRun, read_observer_snapshot

_OBSERVER_ACTIONS = RunActionAvailability(
    cancel_reason="Observer mode is read-only.",
    guidance_reason="Observer mode is read-only.",
    force_stop_reason="Observer mode is read-only.",
)


_LOG_TAIL_BYTES = 2000


def _tail_open_file(log_file: BinaryIO, size: int) -> str:
    _ = log_file.seek(max(0, size - _LOG_TAIL_BYTES))
    data = log_file.read(_LOG_TAIL_BYTES)
    return data.decode("utf-8", errors="replace")


@dataclass(slots=True)
class WatchSnapshotSource:
    """Build bounded presentation snapshots without opening a writer connection."""

    project_root: Path
    db_path: Path
    limit: int = 50
    _tail_cache: dict[Path, tuple[tuple[int, int, int], tuple[str, ...]]] = field(
        default_factory=dict, init=False
    )

    def snapshot(self) -> ExecutionSnapshot:
        observed = read_observer_snapshot(self.db_path, self.limit)
        runs = observed.runs
        active = tuple(
            self._active_snapshot(run, run.description) for run in runs if run.status == "running"
        )
        terminal = tuple(
            self._terminal_snapshot(run, run.description)
            for run in reversed(runs)
            if run.status != "running"
        )
        return ExecutionSnapshot(
            goal=observed.goal or str(self.project_root),
            active_runs=active,
            terminal_runs=terminal,
            completed=sum(run.status == "done" for run in runs),
            failed=sum(run.status == "failed" for run in runs),
            stopped=0,
            available=observed.available,
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
            pending_guidance=None,
            elapsed_seconds=self._duration(run.started_at, None),
            progress_pct=None,
            eta_seconds=None,
            attempt=None,
            max_attempts=None,
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
            pending_guidance=None,
            duration_seconds=self._duration(run.started_at, run.ended_at),
        )

    def _output(self, run: DurableRun) -> tuple[str, ...]:
        candidate = self._log_file(run.log_path)
        if candidate is None:
            return ()
        descriptor = -1
        try:
            selected = os.lstat(candidate)
            if not stat.S_ISREG(selected.st_mode):
                return ()
            descriptor = os.open(candidate, os.O_RDONLY)
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode) or not os.path.samestat(selected, metadata):
                return ()
            signature = (metadata.st_ino, metadata.st_size, metadata.st_mtime_ns)
            cached = self._tail_cache.get(candidate)
            if cached is not None and cached[0] == signature:
                return cached[1]
            with os.fdopen(descriptor, "rb") as log_file:
                descriptor = -1
                output = tuple(_tail_open_file(log_file, metadata.st_size).splitlines())
        except OSError:
            return ()
        finally:
            if descriptor >= 0:
                os.close(descriptor)
        if len(self._tail_cache) >= self.limit and candidate not in self._tail_cache:
            _ = self._tail_cache.pop(next(iter(self._tail_cache)))
        self._tail_cache[candidate] = (signature, output)
        return output

    def _log_file(self, log_path: str) -> Path | None:
        root = self.project_root.resolve()
        candidate = Path(log_path).resolve()
        if not candidate.is_relative_to(root):
            return None
        if candidate.is_file():
            return candidate
        if not candidate.is_dir():
            return None
        for child in reversed(sorted(candidate.glob("*.log"))):
            resolved = child.resolve()
            if resolved.is_file() and resolved.is_relative_to(root):
                return resolved
        return None

    @staticmethod
    def _duration(started_at: str, ended_at: str | None) -> float:
        started = datetime.fromisoformat(started_at)
        ended = datetime.fromisoformat(ended_at) if ended_at else datetime.now(UTC)
        return max(0.0, (ended - started).total_seconds())
