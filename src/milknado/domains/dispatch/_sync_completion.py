"""Completion of a synchronous worker with execution evidence."""

from __future__ import annotations

import logging
from typing import Protocol

from milknado.domains.common import NodeStatus, RunFenceLostError, RunResult
from milknado.domains.dispatch._attempt_completion import (
    AttemptRef,
    settle_counted_failure,
    settle_node_terminal,
)
from milknado.domains.dispatch._runstate import now_iso
from milknado.domains.graph import FinalizeTaskFailure, MikadoGraph, StartTaskAttempt

_logger = logging.getLogger(__name__)


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
    ref = AttemptRef(request.node_id, run_id, run_id, run_id)
    if attempt.start_state == "executed":
        failure = FinalizeTaskFailure(request.node_id, run_id, run_id, run_id, result)
        if settle_counted_failure(graph, failure) == "unknown_start":
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
    graph.task_attempts.require_nonfailure(ref.attempt_id, ref.owner_run_id, outcome)
    try:
        graph.runs.finish(run_id, result)
    except RunFenceLostError:
        _logger.info("sync dispatch failure run already finalized: run_id=%s", run_id)
    _ = settle_node_terminal(graph, ref, NodeStatus.FAILED)
