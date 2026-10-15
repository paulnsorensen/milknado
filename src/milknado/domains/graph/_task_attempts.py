"""Durable, fenced logical task attempts."""

from __future__ import annotations

import sqlite3
from contextlib import AbstractContextManager
from typing import Protocol, cast

from milknado.domains.common import NodeStatus
from milknado.domains.graph._run_persistence import finish_run
from milknado.domains.graph._sqlite_rows import fetchone
from milknado.domains.graph._task_attempt_models import (
    FailureDisposition,
    FinalizeTaskFailure,
    StartTaskAttempt,
    TaskAttemptRecord,
)
from milknado.domains.graph._task_attempt_reads import failed_count, get_attempt


class AttemptGraphPort(Protocol):
    @property
    def synchronization_lock(self) -> AbstractContextManager[object]: ...

    @property
    def group_connection(self) -> sqlite3.Connection: ...

    def group_notifications(self) -> AbstractContextManager[None]: ...

    def release(self, node_id: int, run_id: str) -> bool: ...

    def mark_terminal(self, node_id: int, run_id: str, status: NodeStatus) -> bool: ...

    def mark_blocked_fenced(self, node_id: int, run_id: str) -> bool: ...


class LostAttemptFence(Exception):
    pass


def start_attempt(conn: sqlite3.Connection, request: StartTaskAttempt) -> TaskAttemptRecord:
    if not request.attempt_id or not request.owner_run_id:
        raise ValueError("task attempt and owner identities must be nonempty")
    if type(request.max_node_attempts) is not int or request.max_node_attempts < 1:
        raise ValueError("max_node_attempts must be positive")
    existing = get_attempt(conn, request.attempt_id)
    if existing is not None:
        if (existing.node_id, existing.owner_run_id, existing.max_node_attempts) != (
            request.node_id,
            request.owner_run_id,
            request.max_node_attempts,
        ):
            raise ValueError("task attempt identity belongs to another owner")
        return existing
    owner = fetchone(
        conn,
        "SELECT 1 FROM nodes WHERE id = ? AND status = 'running' AND run_id = ?",
        (request.node_id, request.owner_run_id),
    )
    if owner is None:
        raise ValueError("task attempt owner fence lost")
    _ = conn.execute(
        "INSERT INTO task_attempts "
        + "(attempt_id, node_id, owner_run_id, active_invocation_id, max_node_attempts) "
        + "VALUES (?, ?, ?, ?, ?)",
        (
            request.attempt_id,
            request.node_id,
            request.owner_run_id,
            request.invocation_id or request.owner_run_id,
            request.max_node_attempts,
        ),
    )
    conn.commit()
    record = get_attempt(conn, request.attempt_id)
    assert record is not None
    return record


def record_start_evidence(
    conn: sqlite3.Connection,
    identity: tuple[str, str, str],
    state: str,
) -> bool:
    attempt_id, owner_run_id, invocation_id = identity
    if state not in {"released", "executed", "startup_failed"}:
        raise ValueError("invalid task start evidence")
    allowed = {
        "released": ("unknown",),
        "executed": ("unknown", "released", "executed"),
        "startup_failed": ("unknown", "released", "startup_failed"),
    }[state]
    fence = (attempt_id, owner_run_id, invocation_id, owner_run_id)
    current = fetchone(
        conn,
        "SELECT start_state FROM task_attempts WHERE attempt_id = ? "
        + "AND owner_run_id = ? AND active_invocation_id = ? AND outcome = 'active' "
        + "AND EXISTS (SELECT 1 FROM nodes WHERE nodes.id = task_attempts.node_id "
        + "AND nodes.status = 'running' AND nodes.run_id = ?)",
        fence,
    )
    if current is None or current[0] not in allowed:
        return False
    if current[0] == state:
        return True
    cursor = conn.execute(
        "UPDATE task_attempts SET start_state = ? WHERE attempt_id = ? "
        + "AND owner_run_id = ? AND active_invocation_id = ? AND outcome = 'active' "
        + "AND start_state IN ("
        + ", ".join("?" for _ in allowed)
        + ") "
        + "AND EXISTS (SELECT 1 FROM nodes WHERE nodes.id = task_attempts.node_id "
        + "AND nodes.status = 'running' AND nodes.run_id = ?)",
        (state, attempt_id, owner_run_id, invocation_id, *allowed, owner_run_id),
    )
    conn.commit()
    return cursor.rowcount == 1


def bind_invocation(
    conn: sqlite3.Connection, attempt_id: str, owner_run_id: str, invocation_id: str
) -> bool:
    if not invocation_id:
        raise ValueError("invocation identity must be nonempty")
    current = get_attempt(conn, attempt_id)
    if current is None or current.outcome != "active":
        return False
    if (current.owner_run_id, current.active_invocation_id) == (owner_run_id, invocation_id):
        return True
    cursor = conn.execute(
        "UPDATE task_attempts SET owner_run_id = ?, active_invocation_id = ?, "
        + "start_state = CASE WHEN active_invocation_id = ? THEN start_state "
        + "ELSE 'unknown' END WHERE attempt_id = ? AND outcome = 'active' "
        + "AND EXISTS (SELECT 1 FROM nodes WHERE nodes.id = task_attempts.node_id "
        + "AND nodes.status = 'running' AND nodes.run_id = ?)",
        (owner_run_id, invocation_id, invocation_id, attempt_id, owner_run_id),
    )
    conn.commit()
    return cursor.rowcount == 1


def _group_failure(
    conn: sqlite3.Connection, request: FinalizeTaskFailure, status: NodeStatus
) -> None:
    if request.group_id is None:
        return
    cursor = conn.execute(
        "UPDATE execution_groups SET active_node_id = NULL, active_run_id = NULL, "
        + "active_attempt_id = NULL, active_node_status = NULL, active_node_run_id = NULL "
        + "WHERE id = ? AND active_attempt_id = ? AND active_node_id = ?",
        (request.group_id, request.attempt_id, request.node_id),
    )
    if cursor.rowcount != 1:
        raise LostAttemptFence
    if status is not NodeStatus.PENDING:
        cursor = conn.execute(
            "UPDATE execution_group_tasks SET status = ?, result = ? "
            + "WHERE group_id = ? AND node_id = ? AND status IS NULL",
            (status.value, request.group_result, request.group_id, request.node_id),
        )
        if cursor.rowcount != 1:
            raise LostAttemptFence


def _failure_record(
    conn: sqlite3.Connection, request: FinalizeTaskFailure
) -> tuple[TaskAttemptRecord, FailureDisposition | None]:
    record = get_attempt(conn, request.attempt_id)
    if record is None or record.node_id != request.node_id:
        raise ValueError("task attempt does not belong to node")
    if record.max_node_attempts != request.max_node_attempts:
        raise ValueError("task attempt budget differs from persisted limit")
    if record.outcome == "failed":
        assert record.disposition is not None and record.failed_count is not None
        return record, FailureDisposition(
            NodeStatus(record.disposition), record.failed_count, record.reason or ""
        )
    node_id = request.node_id
    if record.outcome == "startup_failed":
        reason = record.reason or "task execution did not start"
        return record, FailureDisposition("startup_failed", failed_count(conn, node_id), reason)
    if record.outcome != "active" or record.owner_run_id != request.owner_run_id:
        reason = "task attempt is no longer active"
        return record, FailureDisposition("lost_fence", failed_count(conn, node_id), reason)
    query = "SELECT 1 FROM nodes WHERE id = ? AND status = 'running' AND run_id = ?"
    if fetchone(conn, query, (node_id, request.owner_run_id)) is None:
        reason = "task attempt owner fence lost"
        return record, FailureDisposition("lost_fence", failed_count(conn, node_id), reason)
    if request.max_node_attempts > 1 and record.start_state not in {"executed", "startup_failed"}:
        reason = "task execution start is not confirmed"
        return record, FailureDisposition("unknown_start", failed_count(conn, node_id), reason)
    return record, None


def _finish_startup_failure(
    graph: AttemptGraphPort,
    conn: sqlite3.Connection,
    request: FinalizeTaskFailure,
    run_status: str,
) -> FailureDisposition:
    if not graph.mark_terminal(request.node_id, request.owner_run_id, NodeStatus.FAILED):
        raise LostAttemptFence
    if run_status == "running":
        finish_run(conn, request.run_id, request.result)
    _group_failure(conn, request, NodeStatus.FAILED)
    reason = request.result.error or request.result.detail or "task execution did not start"
    _ = conn.execute(
        "UPDATE task_attempts SET outcome = 'startup_failed', reason = ? "
        + "WHERE attempt_id = ? AND outcome = 'active'",
        (reason, request.attempt_id),
    )
    return FailureDisposition("startup_failed", failed_count(conn, request.node_id), reason)


def _finish_counted_failure(
    graph: AttemptGraphPort,
    conn: sqlite3.Connection,
    request: FinalizeTaskFailure,
    run_status: str,
) -> FailureDisposition:
    count = failed_count(conn, request.node_id) + 1
    status = (
        NodeStatus.FAILED
        if request.max_node_attempts == 1
        else NodeStatus.PENDING
        if count < request.max_node_attempts
        else NodeStatus.BLOCKED
    )
    landed = (
        graph.release(request.node_id, request.owner_run_id)
        if status is NodeStatus.PENDING
        else graph.mark_blocked_fenced(request.node_id, request.owner_run_id)
        if status is NodeStatus.BLOCKED
        else graph.mark_terminal(request.node_id, request.owner_run_id, NodeStatus.FAILED)
    )
    if not landed:
        raise LostAttemptFence
    if run_status == "running":
        finish_run(conn, request.run_id, request.result)
    reason = f"attempt {count}/{request.max_node_attempts}: " + (
        request.result.error or request.result.detail or "node failed"
    )
    _group_failure(conn, request, status)
    _ = conn.execute(
        "UPDATE task_attempts SET outcome = 'failed', disposition = ?, "
        + "failed_count = ?, reason = ? WHERE attempt_id = ? AND outcome = 'active'",
        (status.value, count, reason, request.attempt_id),
    )
    return FailureDisposition(status, count, reason)


def _finish_failure(
    graph: AttemptGraphPort,
    conn: sqlite3.Connection,
    request: FinalizeTaskFailure,
) -> FailureDisposition:
    record, prior = _failure_record(conn, request)
    if prior is not None:
        return prior
    if request.result.status != "failed":
        raise ValueError("failure result must be failed")
    query = "SELECT status FROM runs WHERE run_id = ? AND node_id = ? AND attempt_id = ?"
    run = fetchone(conn, query, (request.run_id, request.node_id, request.attempt_id))
    if run is None or run[0] not in {"running", "failed"}:
        raise LostAttemptFence
    run_status = cast(str, run[0])
    if record.start_state == "startup_failed":
        return _finish_startup_failure(graph, conn, request, run_status)
    return _finish_counted_failure(graph, conn, request, run_status)


def finalize_failure(graph: AttemptGraphPort, request: FinalizeTaskFailure) -> FailureDisposition:
    if type(request.max_node_attempts) is not int or request.max_node_attempts < 1:
        raise ValueError("max_node_attempts must be positive")
    with graph.synchronization_lock:
        conn = graph.group_connection
        try:
            with graph.group_notifications(), conn:
                _ = conn.execute("BEGIN IMMEDIATE")
                return _finish_failure(graph, conn, request)
        except LostAttemptFence:
            return FailureDisposition(
                "lost_fence", failed_count(conn, request.node_id), "task attempt owner fence lost"
            )
