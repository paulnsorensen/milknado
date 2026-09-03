"""Data types for run configuration and state.

These are the core types shared across the engine, CLI, manager, and UI
modules.  They are intentionally separate from ``engine.py`` so modules
that only need the types don't pull in the engine's execution logic.
"""

from __future__ import annotations

import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path

from milknado.loop._events import STOP_COMPLETED, STOP_ERROR, STOP_USER_REQUESTED, StopReason

DEFAULT_COMMAND_TIMEOUT: float = 60
"""Default timeout in seconds for commands defined in RALPH.md frontmatter."""

DEFAULT_COMPLETION_SIGNAL = "RALPH_PROMISE_COMPLETE"
"""Default inner ``<promise>...</promise>`` text that marks promise completion."""

RUN_ID_LENGTH: int = 12
"""Number of hex characters used for generated run IDs."""


def generate_run_id() -> str:
    """Generate a short hex run ID from a random UUID."""
    return uuid.uuid4().hex[:RUN_ID_LENGTH]


class RunStatus(Enum):
    """Lifecycle status of a run.

    Transitions follow a simple path: ``PENDING`` → ``RUNNING`` →
    terminal (``COMPLETED``, ``STOPPED``, or ``FAILED``).
    """

    PENDING = "pending"
    RUNNING = "running"
    STOPPED = "stopped"
    COMPLETED = "completed"
    FAILED = "failed"

    @property
    def reason(self) -> StopReason:
        """Return the reason string for terminal statuses.

        Used in ``RUN_STOPPED`` event data.  Only valid for terminal
        statuses (``COMPLETED``, ``FAILED``, ``STOPPED``).

        Raises :class:`ValueError` for non-terminal statuses.
        """
        reason = _STATUS_REASONS.get(self)
        if reason is None:
            raise ValueError(f"{self.name} is not a terminal status")
        return reason


# Maps terminal run statuses to the reason string emitted in RUN_STOPPED events.
_STATUS_REASONS: dict[RunStatus, StopReason] = {
    RunStatus.COMPLETED: STOP_COMPLETED,
    RunStatus.FAILED: STOP_ERROR,
    RunStatus.STOPPED: STOP_USER_REQUESTED,
}


@dataclass(slots=True)
class Command:
    """A named command from RALPH.md frontmatter."""

    name: str
    run: str
    timeout: float = DEFAULT_COMMAND_TIMEOUT


@dataclass(frozen=True, slots=True)
class CompletionVerdict:
    """Tier-2 completion check result returned by a ``completion_verifier``.

    ``ok=True`` accepts the completion; ``ok=False`` rejects it and
    ``feedback`` is injected into the next iteration's context.
    """

    ok: bool
    feedback: str


@dataclass(slots=True)
class RunConfig:
    """All settings for a single run.

    Mutable by design: the engine reads fields at each iteration boundary,
    so you can change ``max_iterations``, ``delay``, or ``timeout`` while
    the loop is running and the new values take effect on the next cycle.
    """

    agent: str
    ralph_dir: Path
    ralph_file: Path | None = None
    # In-memory prompt *body* (no frontmatter). Mutually exclusive with
    # ``ralph_file``: supply exactly one. Placeholders are still resolved.
    prompt: str | None = None
    commands: list[Command] = field(default_factory=list)
    args: dict[str, str] = field(default_factory=dict)
    max_iterations: int | None = None
    delay: float = 0
    timeout: float | None = None
    stop_on_error: bool = False
    # End the run as FAILED once this many iterations fail in a row; ``None``
    # disables the cap. Guards against a permanently-broken agent command
    # (e.g. an unknown CLI flag) crash-looping forever.
    max_consecutive_failures: int | None = None
    log_dir: Path | None = None
    project_root: Path = field(default=Path("."))
    commit_footer: str | None = None
    # Inner text expected inside ``<promise>...</promise>``.
    completion_signal: str = DEFAULT_COMPLETION_SIGNAL
    # Stop the run when the configured promise payload is observed.
    stop_on_completion_signal: bool = False
    # Per-iteration tool-use cap; None disables the cap.
    max_turns: int | None = None
    # Soft wind-down fires at ``max_turns - max_turns_grace``.
    max_turns_grace: int = 2
    # Tier-2 completion check consulted when exit-0 + promise would
    # complete the run; ``None`` accepts on the promise alone.
    completion_verifier: Callable[[], CompletionVerdict] | None = None
    # Optional durable completion probe evaluated after a successful agent turn.
    # Unlike ``completion_verifier``, it can establish completion without a
    # promise tag when an external system commits the terminal signal.
    completion_probe: Callable[[], bool] | None = None

    def __post_init__(self) -> None:
        if (self.prompt is None) == (self.ralph_file is None):
            raise ValueError("RunConfig requires exactly one of `prompt` or `ralph_file`")


@dataclass(slots=True)
class RunState:
    """Observable state for a run.

    Stop methods use :class:`threading.Event` so the run loop can react
    without busy-waiting.

    **Counter invariant**: ``timed_out_count`` is a *subset* of ``failed``,
    not an independent category.  A timed-out iteration increments both
    ``timed_out_count`` and ``failed``.  Therefore
    ``completed + failed == total iterations`` (use :attr:`total`).
    """

    run_id: str
    status: RunStatus = RunStatus.PENDING
    iteration: int = 0
    completed: int = 0
    failed: int = 0
    timed_out_count: int = 0
    # Failed iterations since the last completed one; reset by mark_completed.
    consecutive_failures: int = 0
    started_at: datetime | None = None
    # Last agent turn output, retained so coordinators can capture a resumable
    # session id before the worker run is replaced by a review round.
    last_result_text: str | None = None
    last_captured_stdout: str | None = None
    last_captured_stderr: str | None = None
    promise_completed: bool = False

    _stop_event: threading.Event = field(
        default_factory=threading.Event, init=False, repr=False, compare=False
    )
    _force_stop_event: threading.Event = field(
        default_factory=threading.Event, init=False, repr=False, compare=False
    )
    _guidance: list[str] = field(default_factory=list, init=False, repr=False, compare=False)
    _guidance_open: bool = field(default=True, init=False, repr=False, compare=False)
    _guidance_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False, compare=False
    )

    def queue_guidance(self, text: str) -> bool:
        """Queue non-empty operator guidance for one later prompt."""
        if not text.strip():
            return False
        with self._guidance_lock:
            if not self._guidance_open:
                return False
            self._guidance.append(text)
            return True

    def take_guidance(self) -> tuple[str, ...]:
        """Consume the guidance admitted for the next prompt exactly once."""
        with self._guidance_lock:
            guidance = tuple(self._guidance)
            self._guidance.clear()
            return guidance

    @property
    def pending_guidance(self) -> tuple[str, ...]:
        """Return a stable view of guidance awaiting the next agent prompt."""
        with self._guidance_lock:
            return tuple(self._guidance)

    def _try_commit_completion(self, *, require_empty_guidance: bool) -> bool:
        if self._stop_event.is_set() or self._force_stop_event.is_set():
            self.status = RunStatus.STOPPED
            return False
        if require_empty_guidance and self._guidance:
            return False
        self._guidance_open = False
        self.status = RunStatus.COMPLETED
        return True

    def try_commit_soft_completion(self) -> bool:
        """Atomically commit soft completion unless guidance or a stop won first."""
        with self._guidance_lock:
            return self._try_commit_completion(require_empty_guidance=True)

    def try_commit_completion(self) -> bool:
        """Atomically commit terminal completion unless a stop won first."""
        with self._guidance_lock:
            return self._try_commit_completion(require_empty_guidance=False)

    def try_commit_failure(self) -> bool:
        """Atomically commit failure unless a stop won first."""
        with self._guidance_lock:
            if self._stop_event.is_set() or self._force_stop_event.is_set():
                self.status = RunStatus.STOPPED
                return False
            self._guidance_open = False
            self.status = RunStatus.FAILED
            return True

    def close_guidance(self) -> None:
        """Reject later guidance while retaining accepted undelivered text."""
        with self._guidance_lock:
            self._guidance_open = False

    @property
    def force_stop_requested(self) -> bool:
        return self._force_stop_event.is_set()

    def request_force_stop(self) -> None:
        with self._guidance_lock:
            self._guidance_open = False
            self._force_stop_event.set()
            self._stop_event.set()

    @property
    def force_stop_event(self) -> threading.Event:
        return self._force_stop_event

    @property
    def total(self) -> int:
        """Total iterations run (``completed + failed``)."""
        return self.completed + self.failed

    def request_stop(self) -> None:
        with self._guidance_lock:
            self._guidance_open = False
            self._stop_event.set()

    @property
    def stop_requested(self) -> bool:
        return self._stop_event.is_set()

    def wait_for_stop(self, timeout: float | None = None) -> bool:
        """Block until a stop is requested or timeout. Returns True if stopped."""
        return self._stop_event.wait(timeout=timeout)

    def mark_completed(self) -> None:
        self.completed += 1
        self.consecutive_failures = 0

    def mark_failed(self) -> None:
        self.failed += 1
        self.consecutive_failures += 1

    def mark_timed_out(self) -> None:
        """Record a timed-out iteration (also counts as failed)."""
        self.timed_out_count += 1
        self.mark_failed()
