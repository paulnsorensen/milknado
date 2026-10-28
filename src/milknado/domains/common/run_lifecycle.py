"""Run lifecycle values shared by the loop engine and the execution controller."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

StopReason = Literal["completed", "error", "user_requested"]
"""Valid reason strings for run-stopped events."""


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


_STATUS_REASONS: dict[RunStatus, StopReason] = {
    RunStatus.COMPLETED: "completed",
    RunStatus.FAILED: "error",
    RunStatus.STOPPED: "user_requested",
}


@dataclass(frozen=True, slots=True)
class CompletionVerdict:
    """Tier-2 completion check result returned by a ``completion_verifier``.

    ``ok=True`` accepts the completion; ``ok=False`` rejects it and
    ``feedback`` is injected into the next iteration's context.
    """

    ok: bool
    feedback: str
