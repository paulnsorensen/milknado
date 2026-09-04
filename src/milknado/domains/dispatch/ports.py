from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from milknado.domains.common import NodeStatus, RunResult
from milknado.domains.common.config import MilknadoConfig
from milknado.domains.graph import MikadoGraph


@dataclass(frozen=True)
class RunWindow:
    run_id: str
    argv: tuple[str, ...]
    cwd: Path
    log_path: Path
    exit_code_path: Path
    env: dict[str, str] = field(default_factory=dict)
    brief_path: Path | None = None


class FinishDispatchPort(Protocol):
    """The graph capability required to persist a worker terminal result."""

    def finish_run(self, run_id: str, result: RunResult, /) -> bool: ...
    def mark_terminal(self, node_id: int, run_id: str, status: NodeStatus, /) -> bool: ...
    def get_node(self, node_id: int, /) -> object | None: ...


class RunReaderPort(Protocol):
    """The graph capability required to observe one persisted run."""

    def get_run(self, run_id: str, /) -> Mapping[str, object] | None: ...


class RunFinalizerPort(RunReaderPort, Protocol):
    """The graph capabilities required to finalize one persisted run."""

    def finish_run(self, run_id: str, result: RunResult, /) -> bool: ...


class StaleRunPort(Protocol):
    """The graph capability required by stale-run reconciliation."""

    def runs_for_node(self, node_id: int, /) -> Iterable[Mapping[str, object]]: ...
    def finish_run(self, run_id: str, result: RunResult, /) -> bool: ...


class WorkerOutcomePort(Protocol):
    """The worker result fields consumed by lifecycle finalization."""

    @property
    def exit_code(self) -> int: ...

    @property
    def timed_out(self) -> bool: ...


class TmuxPort(Protocol):
    def available(self) -> bool: ...
    def ensure_session(self) -> None: ...
    def window_exists(self, run_id: str) -> bool: ...
    def target_for(self, run_id: str) -> str: ...
    def open_run_window(self, window: RunWindow) -> int: ...
    def kill_window(self, run_id: str) -> bool: ...


@dataclass(frozen=True)
class ProcessOutcome:
    exit_code: int
    timed_out: bool
    cancelled: bool


class ProcessPort(Protocol):
    def run(
        self,
        argv: tuple[str, ...],
        cwd: Path,
        log_path: Path,
        stdin: bytes,
        env: dict[str, str],
        timeout: float,
        *,
        cancel_requested: Callable[[], bool] | None = None,
        on_started: Callable[[int], None] | None = None,
    ) -> ProcessOutcome: ...

    def spawn_detached(
        self,
        argv: tuple[str, ...],
        cwd: Path,
        log_path: Path,
        env: dict[str, str],
    ) -> int: ...

    def terminate_group(self, pid: int, timeout: float) -> bool: ...


class ProcessTerminationPort(Protocol):
    def terminate_group(self, pid: int, timeout: float) -> bool:
        """Request termination, escalate if needed, and return only after exit."""
        ...


class GraphSessionPort(Protocol):
    def open_graph(self, project_root: Path) -> tuple[MikadoGraph, MilknadoConfig]: ...
