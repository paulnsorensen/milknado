"""Read-only projection of durable graph and run state for observers."""

from __future__ import annotations

import os
import sqlite3
import stat
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO, Protocol

from milknado.app.run import (
    ActiveRunSnapshot,
    ExecutionRunStatus,
    ExecutionSnapshot,
    RunActionAvailability,
    TerminalRunSnapshot,
)
from milknado.app.run_source import NodeSnapshotRequest
from milknado.domains.common import SessionInput
from milknado.domains.graph import (
    DurableRun,
    GraphSnapshot,
    MikadoGraph,
    NodeDetailResponse,
    admit_session_command,
    connect_readonly,
    read_observer_node_snapshot,
    read_observer_snapshot,
)


class _AttachedSnapshotSource(Protocol):
    def snapshot(self, request: NodeSnapshotRequest | None = None) -> ExecutionSnapshot: ...

    def node_snapshot(self, request: NodeSnapshotRequest) -> NodeDetailResponse: ...

    def subscribe(self, listener: Callable[[ExecutionSnapshot], None]) -> Callable[[], None]: ...


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
    _graph_connection: sqlite3.Connection | None = field(default=None, init=False)
    _graph_cache: GraphSnapshot | None = field(default=None, init=False)
    _graph_revision: int | None = field(default=None, init=False)

    def snapshot(self, request: NodeSnapshotRequest | None = None) -> ExecutionSnapshot:
        if self._graph_connection is None:
            self._graph_connection = connect_readonly(self.db_path)
        observed = read_observer_snapshot(
            self.db_path,
            self.limit,
            node_id=request.node_id if request is not None else None,
            request_generation=request.request_generation if request is not None else 0,
            page=request.page if request is not None else 0,
            node_limit=request.limit if request is not None else 50,
            session_event_page=request.session_event_page if request is not None else 0,
            cached_graph=self._graph_cache,
            cached_graph_revision=self._graph_revision,
            connection=self._graph_connection,
        )
        self._graph_cache = observed.graph
        self._graph_revision = observed.graph_revision
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
            graph=observed.graph,
            node=observed.node,
        )

    def node_snapshot(  # noqa: V105 - shared source contract consumed by the watch view
        self, request: NodeSnapshotRequest
    ) -> NodeDetailResponse:
        return read_observer_node_snapshot(
            self.db_path,
            request.node_id,
            request.request_generation,
            request.page,
            request.limit,
            request.session_event_page,
        )

    def close(self) -> None:
        if self._graph_connection is not None:
            self._graph_connection.close()
            self._graph_connection = None

    @staticmethod
    def subscribe(listener: Callable[[ExecutionSnapshot], None]) -> Callable[[], None]:
        """Keep the observer source detached from writer notifications."""
        del listener
        return lambda: None

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
            session=run.session,
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
            session=run.session,
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


def graph_command_admitter(graph: MikadoGraph) -> Callable[[str, SessionInput], bool]:
    """Admit attached-watch input against the current owner capability snapshot."""

    def admit(run_id: str, command: SessionInput) -> bool:
        return admit_session_command(graph, run_id, command) is not None

    return admit


@dataclass(slots=True)
class AttachedWatchSource:
    """Add an explicit owner admission seam to an observer snapshot source."""

    source: _AttachedSnapshotSource
    admit: Callable[[str, SessionInput], bool]

    def snapshot(self, request: NodeSnapshotRequest | None = None) -> ExecutionSnapshot:
        return self.source.snapshot(request)

    def node_snapshot(self, request: NodeSnapshotRequest) -> NodeDetailResponse:
        return self.source.node_snapshot(request)

    def subscribe(self, listener: Callable[[ExecutionSnapshot], None]) -> Callable[[], None]:
        return self.source.subscribe(listener)

    def close(self) -> None:
        close = getattr(self.source, "close", None)
        if callable(close):
            _ = close()

    def session_input(self, run_id: str, command: SessionInput) -> bool:
        return self.admit(run_id, command)
