"""Graph-facing operations for durable task attempts."""

from __future__ import annotations

from milknado.domains.graph._task_attempt_binding import bind_worker_invocation
from milknado.domains.graph._task_attempt_models import (
    FailureDisposition,
    FinalizeTaskFailure,
    StartTaskAttempt,
    TaskAttemptRecord,
)
from milknado.domains.graph._task_attempt_reads import failed_count, get_attempt
from milknado.domains.graph._task_attempts import (
    AttemptGraphPort,
    LostAttemptFence,
    bind_invocation,
    finalize_failure,
    record_start_evidence,
    start_attempt,
)


class TaskAttemptFacade:
    def __init__(self, graph: AttemptGraphPort) -> None:
        self._graph: AttemptGraphPort = graph

    def start(self, request: StartTaskAttempt) -> TaskAttemptRecord:
        with self._graph.synchronization_lock:
            return start_attempt(self._graph.group_connection, request)

    def get(self, attempt_id: str) -> TaskAttemptRecord | None:
        with self._graph.synchronization_lock:
            return get_attempt(self._graph.group_connection, attempt_id)

    def failed_count(self, node_id: int) -> int:
        with self._graph.synchronization_lock:
            return failed_count(self._graph.group_connection, node_id)

    def rotate_review_round(
        self, attempt_id: str, old_owner: str, new_owner: str, invocation_id: str
    ) -> bool:
        if not new_owner or not invocation_id:
            raise ValueError("review owner and invocation must be nonempty")
        with self._graph.synchronization_lock:
            conn = self._graph.group_connection
            try:
                with conn:
                    _ = conn.execute("BEGIN IMMEDIATE")
                    node = conn.execute(
                        "UPDATE nodes SET run_id = ? WHERE id = "
                        + "(SELECT node_id FROM task_attempts WHERE attempt_id = ?) "
                        + "AND status = 'running' AND run_id = ?",
                        (new_owner, attempt_id, old_owner),
                    )
                    if node.rowcount != 1:
                        raise LostAttemptFence
                    attempt = conn.execute(
                        "UPDATE task_attempts SET owner_run_id = ?, active_invocation_id = ?, "
                        + "start_state = 'unknown' WHERE attempt_id = ? AND owner_run_id = ? "
                        + "AND outcome = 'active' AND EXISTS (SELECT 1 FROM runs "
                        + "WHERE runs.run_id = ? AND runs.node_id = task_attempts.node_id "
                        + "AND runs.attempt_id = task_attempts.attempt_id "
                        + "AND runs.status = 'running')",
                        (new_owner, invocation_id, attempt_id, old_owner, invocation_id),
                    )
                    if attempt.rowcount != 1:
                        raise LostAttemptFence
            except LostAttemptFence:
                return False
        return True

    def bind_invocation(self, attempt_id: str, owner_run_id: str, invocation_id: str) -> bool:
        with self._graph.synchronization_lock:
            return bind_invocation(
                self._graph.group_connection, attempt_id, owner_run_id, invocation_id
            )

    def bind_worker_invocation(self, identity: tuple[str, str, str, str]) -> bool:
        with self._graph.synchronization_lock:
            return bind_worker_invocation(self._graph.group_connection, identity)

    def confirm_execution(self, attempt_id: str, owner_run_id: str, invocation_id: str) -> bool:
        with self._graph.synchronization_lock:
            return record_start_evidence(
                self._graph.group_connection,
                (attempt_id, owner_run_id, invocation_id),
                "executed",
            )

    def record_release(self, attempt_id: str, owner_run_id: str, invocation_id: str) -> bool:
        with self._graph.synchronization_lock:
            return record_start_evidence(
                self._graph.group_connection,
                (attempt_id, owner_run_id, invocation_id),
                "released",
            )

    def record_startup_failure(
        self, attempt_id: str, owner_run_id: str, invocation_id: str
    ) -> bool:
        with self._graph.synchronization_lock:
            return record_start_evidence(
                self._graph.group_connection,
                (attempt_id, owner_run_id, invocation_id),
                "startup_failed",
            )

    def record_nonfailure(self, attempt_id: str, owner_run_id: str, outcome: str) -> bool:
        if outcome not in {"done", "cancelled", "review_blocked", "preserved", "startup_failed"}:
            raise ValueError("invalid nonfailure task outcome")
        with self._graph.synchronization_lock:
            conn = self._graph.group_connection
            record = get_attempt(conn, attempt_id)
            if record is None or record.owner_run_id != owner_run_id:
                return False
            if record.outcome == outcome:
                return True
            if record.outcome != "active":
                return False
            cursor = conn.execute(
                "UPDATE task_attempts SET outcome = ? WHERE attempt_id = ? "
                + "AND owner_run_id = ? AND outcome = 'active' "
                + "AND EXISTS (SELECT 1 FROM nodes WHERE nodes.id = task_attempts.node_id "
                + "AND nodes.run_id = ? "
                + "AND nodes.status IN ('running', 'done', 'blocked', 'failed'))",
                (outcome, attempt_id, owner_run_id, owner_run_id),
            )
            conn.commit()
            return cursor.rowcount == 1

    def fail(self, request: FinalizeTaskFailure) -> FailureDisposition:
        return finalize_failure(self._graph, request)
