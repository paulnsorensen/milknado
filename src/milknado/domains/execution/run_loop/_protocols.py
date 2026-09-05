"""Structural view of RunLoop used by run-loop collaborators.

Declared here, in a module neither RunLoop nor its collaborators import from
each other, so completion handling can name the state it reads without an
import edge back to the concrete class.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from milknado.domains.common import ProgressEvent
    from milknado.domains.common.protocols import LoopPort
    from milknado.domains.execution.executor import Executor
    from milknado.domains.execution.run_loop.input import InputState
    from milknado.domains.execution.run_loop.state import TerminalRunState
    from milknado.domains.graph import MikadoGraph


class RunLoopState(Protocol):
    """The RunLoop state surface that completion handling reads and mutates."""

    _active: dict[str, int]
    _attempts: dict[int, int]
    _completion_durations: deque[float]
    _dispatched_at: dict[str, float]
    _executor: Executor
    _failure_triggered: bool
    _graph: MikadoGraph
    _input: InputState
    _logs: deque[str]
    _progress_by_run: dict[str, ProgressEvent]
    _ralph: LoopPort
    _stopped: int
    _stopped_nodes: set[int]
    _strict: bool
    _terminal_runs: deque[TerminalRunState]
