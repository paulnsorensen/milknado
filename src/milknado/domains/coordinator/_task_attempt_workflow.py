"""Native task-attempt run linkage and completion."""

from __future__ import annotations

from datetime import UTC, datetime

from milknado.domains.common import RunResult
from milknado.domains.execution import NodeLoopOutcome
from milknado.domains.graph import (
    FinalizeTaskFailure,
    MikadoGraph,
    StartTaskAttempt,
    TaskAttempt,
    TaskOutcome,
)


def link_task_run(graph: MikadoGraph, attempt: TaskAttempt, max_node_attempts: int) -> None:
    group = graph.groups.get(attempt.group_id)
    if group is None:
        raise ValueError("execution group disappeared during launch")
    if graph.task_attempts.get(attempt.attempt_id) is None:
        _ = graph.task_attempts.start(
            StartTaskAttempt(
                attempt.node_id,
                attempt.attempt_id,
                attempt.attempt_id,
                attempt.attempt_id,
                max_node_attempts,
            )
        )
    if graph.runs.get(attempt.attempt_id) is None:
        graph.runs.start(
            attempt.attempt_id,
            attempt.node_id,
            group.worktree_path,
            datetime.now(UTC).isoformat(),
            None,
            attempt_id=attempt.attempt_id,
        )


def _fail_task(graph: MikadoGraph, attempt: TaskAttempt, outcome: NodeLoopOutcome) -> bool:
    record = graph.task_attempts.get(attempt.attempt_id)
    if record is None:
        raise ValueError("native task attempt is not linked")
    disposition = graph.task_attempts.fail(
        FinalizeTaskFailure(
            attempt.node_id,
            attempt.attempt_id,
            attempt.attempt_id,
            attempt.attempt_id,
            record.max_node_attempts,
            RunResult("failed", None, False, datetime.now(UTC).isoformat(), detail=outcome.detail),
            group_id=attempt.group_id,
            group_result=outcome.detail or "",
        )
    )
    if disposition.status == "lost_fence":
        raise ValueError("native task attempt fence lost")
    return disposition.status != "unknown_start"


def _finish_nonfailure(graph: MikadoGraph, attempt: TaskAttempt, status: str, detail: str) -> None:
    graph.groups.finish_task(attempt, TaskOutcome(status, detail))
    if not graph.task_attempts.record_nonfailure(
        attempt.attempt_id,
        attempt.attempt_id,
        "done" if status == "done" else "preserved",
    ):
        raise ValueError("native task outcome fence lost")


def _replay_failure(
    graph: MikadoGraph, attempt: TaskAttempt, outcome: NodeLoopOutcome, state: str
) -> str | None:
    prior = graph.runs.get(attempt.attempt_id)
    if (
        not outcome.success
        and not outcome.ownership_preserved
        and prior is not None
        and prior["node_id"] == attempt.node_id
        and prior["attempt_id"] == attempt.attempt_id
        and prior["status"] == "failed"
        and prior["detail"] == outcome.detail
    ):
        return None if state == "finished" else "failed"
    raise ValueError("worker result conflicts with recorded task attempt")


def finish_task_attempt(
    graph: MikadoGraph, attempt: TaskAttempt, outcome: NodeLoopOutcome, state: str
) -> str | None:
    status = "done" if outcome.success else "blocked" if outcome.ownership_preserved else "failed"
    detail = outcome.detail or ""
    result = graph.groups.task_result(attempt.node_id)
    record = graph.task_attempts.get(attempt.attempt_id)
    if record is None:
        raise ValueError("native task attempt is not linked")
    if record.node_id != attempt.node_id or record.owner_run_id != attempt.attempt_id:
        raise ValueError("native task attempt owner fence lost")
    if record.outcome == "failed":
        return _replay_failure(graph, attempt, outcome, state)
    if result is None and status == "failed":
        if not _fail_task(graph, attempt, outcome):
            return None
    elif result is None:
        _finish_nonfailure(graph, attempt, status, detail)
    elif result != (status, detail):
        raise ValueError("worker result conflicts with recorded task result")
    elif status != "failed":
        canonical = "done" if status == "done" else "preserved"
        if record.outcome == "active":
            if not graph.task_attempts.record_nonfailure(
                attempt.attempt_id, attempt.attempt_id, canonical
            ):
                raise ValueError("native task outcome fence lost")
        elif record.outcome != canonical:
            raise ValueError("worker result conflicts with recorded task attempt")
    run = graph.runs.get(attempt.attempt_id)
    if run is not None and run["status"] == "running":
        graph.runs.finish(
            attempt.attempt_id,
            RunResult(status, None, False, datetime.now(UTC).isoformat(), detail=outcome.detail),
        )
    return status
