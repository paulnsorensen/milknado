"""Event types and emitter protocol for the run loop.

The run engine emits structured events during execution so that different
frontends can observe progress without coupling to the engine internals.
"""

from __future__ import annotations

import queue
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import (
    Generic,
    Literal,
    NotRequired,
    Protocol,
    TypedDict,
    TypeVar,
    cast,
    runtime_checkable,
)

LogLevel = Literal["info", "error"]
"""Valid log levels for :class:`LogMessageData` events."""

OutputStream = Literal["stdout", "stderr"]
"""Standard stream observed by agent output callbacks."""

LOG_INFO: LogLevel = "info"
LOG_ERROR: LogLevel = "error"

StopReason = Literal["completed", "error", "user_requested"]
"""Valid reason strings for :class:`RunStoppedData` events."""

STOP_COMPLETED: StopReason = "completed"
STOP_ERROR: StopReason = "error"
STOP_USER_REQUESTED: StopReason = "user_requested"


class EventType(Enum):
    """All event types emitted by the run loop.

    Events cover run, iteration, command, prompt, cap, and log state.

    **Run lifecycle** — emitted once per run start and stop:
    ``RUN_STARTED`` and ``RUN_STOPPED``.

    **Iteration lifecycle** — emitted once per iteration:
    ``ITERATION_STARTED``, ``ITERATION_COMPLETED``, ``ITERATION_FAILED``,
    ``ITERATION_TIMED_OUT``.

    **Commands** — emitted around command execution:
    ``COMMANDS_STARTED``, ``COMMANDS_COMPLETED``.

    **Prompt assembly** — emitted after the prompt is built:
    ``PROMPT_ASSEMBLED``.


    The ``LOG_MESSAGE`` type is used for general informational and error
    messages (e.g. delay notifications, crash reports).
    """

    # ── Run lifecycle ───────────────────────────────────────────
    RUN_STARTED = "run_started"
    RUN_STOPPED = "run_stopped"

    # ── Iteration lifecycle ─────────────────────────────────────
    ITERATION_STARTED = "iteration_started"
    ITERATION_COMPLETED = "iteration_completed"
    ITERATION_FAILED = "iteration_failed"
    ITERATION_TIMED_OUT = "iteration_timed_out"

    # ── Commands ────────────────────────────────────────────────
    COMMANDS_STARTED = "commands_started"
    COMMANDS_COMPLETED = "commands_completed"

    # ── Prompt assembly ─────────────────────────────────────────
    PROMPT_ASSEMBLED = "prompt_assembled"

    # ── Turn-cap enforcement ────────────────────────────────────
    ITERATION_TURN_CAPPED = "iteration_turn_capped"

    # ── Other ───────────────────────────────────────────────────
    LOG_MESSAGE = "log_message"


# ── Typed event data payloads ─────────────────────────────────────────


class RunStartedData(TypedDict):
    agent: str
    commands: int
    max_iterations: int | None
    timeout: float | None
    delay: float


class RunStoppedData(TypedDict):
    reason: StopReason
    total: int
    completed: int
    failed: int
    timed_out_count: int


class IterationStartedData(TypedDict):
    iteration: int


class IterationEndedData(TypedDict):
    iteration: int
    returncode: int | None
    duration: float
    detail: str
    log_file: str | None
    result_text: str | None
    echo_stdout: NotRequired[str | None]


class CommandsStartedData(TypedDict):
    iteration: int
    count: int


class CommandsCompletedData(TypedDict):
    iteration: int
    count: int


class PromptAssembledData(TypedDict):
    iteration: int


class TurnCappedData(TypedDict):
    iteration: int
    count: int


class LogMessageData(TypedDict):
    message: str
    level: LogLevel
    traceback: NotRequired[str]


class NoData(TypedDict):
    """Empty payload for events that carry no data."""


EventData = (
    NoData
    | RunStartedData
    | RunStoppedData
    | IterationStartedData
    | IterationEndedData
    | CommandsStartedData
    | CommandsCompletedData
    | PromptAssembledData
    | TurnCappedData
    | LogMessageData
)
"""Union of all typed event data payloads."""


# Plain TypeVar (no PEP 696 default) — the Python floor is 3.11; the
# ``default=`` arg needs 3.13+ or a runtime typing_extensions dep, which
# would fight the pyyaml-only core. Bare ``Event`` references resolve
# to the EventData bound.
DataT = TypeVar("DataT", bound="EventData")


@dataclass(slots=True)
class Event(Generic[DataT]):
    """A structured event emitted by the run loop.

    Generic over its payload so embedders can annotate handlers with the
    concrete data type (``def on(e: Event[IterationEndedData])``) without
    casting at every access site.
    """

    type: EventType
    run_id: str
    data: DataT = field(default_factory=lambda: cast(DataT, cast(object, {})))
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


@runtime_checkable
class EventEmitter(Protocol):
    """Protocol for objects that receive run-loop events."""

    def emit(self, event: Event[EventData]) -> None: ...


class NullEmitter:
    """Discards all events silently."""

    def emit(self, event: Event[EventData]) -> None:
        del event


class QueueEmitter:
    """Pushes events into a :class:`queue.Queue` for async consumption."""

    def __init__(self, q: queue.Queue[Event[EventData]] | None = None) -> None:
        self.queue: queue.Queue[Event[EventData]] = q or queue.Queue()

    def emit(self, event: Event[EventData]) -> None:
        self.queue.put(event)


class BoundEmitter:
    """Wraps an EventEmitter with a fixed run_id for concise emission.

    Instead of constructing :class:`Event` objects manually at every call
    site, callers create a ``BoundEmitter`` once with the run ID and then
    emit events with just the type and optional data payload.
    """

    def __init__(self, emitter: EventEmitter, run_id: str) -> None:
        self._emitter: EventEmitter = emitter
        self._run_id: str = run_id

    def __call__(
        self,
        event_type: EventType,
        data: EventData | None = None,
    ) -> None:
        event: Event[EventData] = Event(
            type=event_type,
            run_id=self._run_id,
            data=data if data is not None else NoData(),
        )
        self._emitter.emit(event)

    def log_info(self, message: str) -> None:
        """Emit a ``LOG_MESSAGE`` event at info level."""
        self(EventType.LOG_MESSAGE, LogMessageData(message=message, level=LOG_INFO))

    def log_error(self, message: str, *, traceback: str | None = None) -> None:
        """Emit a ``LOG_MESSAGE`` event at error level."""
        data = LogMessageData(message=message, level=LOG_ERROR)
        if traceback is not None:
            data["traceback"] = traceback
        self(EventType.LOG_MESSAGE, data)
