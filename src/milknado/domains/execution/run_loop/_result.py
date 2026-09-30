from __future__ import annotations

from dataclasses import dataclass

from milknado.domains.execution._models import RebaseConflict


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


@dataclass(frozen=True)
class NodeLoopOutcome:
    node_id: int
    success: bool
    detail: str | None = None
    timed_out: bool = False
    ownership_preserved: bool = False
    worker_run_id: str | None = None
    owner_run_id: str | None = None
