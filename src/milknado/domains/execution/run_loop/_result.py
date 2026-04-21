from __future__ import annotations

from dataclasses import dataclass

from milknado.domains.execution.executor import RebaseConflict


@dataclass(frozen=True)
class VerifyOutcome:
    done: bool
    goal_delta: str | None = None


@dataclass(frozen=True)
class RunLoopResult:
    root_done: bool
    dispatched_total: int
    completed_total: int
    failed_total: int
    rebase_conflicts: tuple[RebaseConflict, ...] = ()
    strict_exit: bool = False
    verify_outcome: VerifyOutcome | None = None
