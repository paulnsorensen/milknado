"""Completion of a synchronous worker with execution evidence."""

from __future__ import annotations

from typing import Protocol

from milknado.domains.common import NodeStatus, RunResult
from milknado.domains.dispatch._runstate import now_iso
from milknado.domains.graph import FinalizeTaskFailure, MikadoGraph, StartTaskAttempt


class SyncFailureRequest(Protocol):
    @property
    def node_id(self) -> int: ...

    @property
    def max_node_attempts(self) -> int: ...

    @property
    def timeout_seconds(self) -> int: ...


def start_sync_attempt(
    graph: MikadoGraph, request: SyncFailureRequest, run_id: str, log_path: str
) -> None:
    _ = graph.task_attempts.start(
        StartTaskAttempt(request.node_id, run_id, run_id, run_id, request.max_node_attempts)
    )
    graph.runs.start(
        run_id,
        request.node_id,
        log_path,
        now_iso(),
        request.timeout_seconds,
        attempt_id=run_id,
    )


def confirm_execution(graph: MikadoGraph, run_id: str, pid: int) -> None:
    graph.runs.set_pid(run_id, pid)
    if not graph.task_attempts.confirm_execution(run_id, run_id, run_id):
        raise RuntimeError(f"task execution proof fence lost for run {run_id}")


def finish_attempt_exception(
    graph: MikadoGraph,
    request: SyncFailureRequest,
    identity: tuple[str, bool],
    exc: Exception,
) -> None:
    run_id, direct_process = identity
    if graph.runs.live_workers(run_id=run_id):
        return
    attempt = graph.task_attempts.get(run_id)
    if attempt is None:
        raise ValueError(f"task attempt missing for run {run_id}")
    result = RunResult("failed", -1, False, now_iso(), error=f"{type(exc).__name__}: {exc}")
    if attempt.start_state == "executed":
        disposition = graph.task_attempts.fail(
            FinalizeTaskFailure(
                request.node_id, run_id, run_id, run_id, request.max_node_attempts, result
            )
        )
        if disposition.status in {"lost_fence", "unknown_start"}:
            raise RuntimeError(f"task attempt fence lost for run {run_id}")
        return
    if direct_process and isinstance(exc, OSError):
        if not graph.task_attempts.record_startup_failure(run_id, run_id, run_id):
            raise RuntimeError(f"startup evidence fence lost for run {run_id}")
        outcome = "startup_failed"
    elif request.max_node_attempts > 1:
        return
    else:
        outcome = "preserved"
    if not graph.task_attempts.record_nonfailure(run_id, run_id, outcome):
        raise RuntimeError(f"task outcome fence lost for run {run_id}")
    graph.runs.finish(run_id, result)
    if not graph.mark_terminal(request.node_id, run_id, NodeStatus.FAILED):
        raise RuntimeError(f"terminal node fence lost for run {run_id}")


def finish_sync_exception(
    graph: MikadoGraph, request: SyncFailureRequest, run_id: str, exc: Exception
) -> None:
    finish_attempt_exception(graph, request, (run_id, True), exc)
