"""Graph-facing operations for durable task attempts."""

from __future__ import annotations

import logging

from milknado.domains.graph._task_attempt_binding import bind_worker_invocation
from milknado.domains.graph._task_attempt_models import (
    FailureDisposition,
    FinalizeTaskFailure,
    StartTaskAttempt,
    TaskAttemptRecord,
)
from milknado.domains.graph._task_attempt_outcomes import record_nonfailure, swap_owner
from milknado.domains.graph._task_attempt_reads import failed_count, get_attempt
from milknado.domains.graph._task_attempts import (
    AttemptGraphPort,
    bind_invocation,
    finalize_failure,
    record_start_evidence,
    start_attempt,
)

_logger = logging.getLogger(__name__)


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
            ids = (attempt_id, old_owner, new_owner, invocation_id)
            return swap_owner(self._graph.group_connection, ids, review_round=True)

    def adopt_owner(self, attempt_id: str, expected_run_id: str, run_id: str) -> bool:
        """Swap the provisional node fence and bind the attempt to the new run atomically."""
        with self._graph.synchronization_lock:
            ids = (attempt_id, expected_run_id, run_id, run_id)
            return swap_owner(self._graph.group_connection, ids, review_round=False)

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
        with self._graph.synchronization_lock:
            return record_nonfailure(
                self._graph.group_connection, attempt_id, owner_run_id, outcome
            )

    def require_nonfailure(self, attempt_id: str, owner_run_id: str, outcome: str) -> None:
        if not self.record_nonfailure(attempt_id, owner_run_id, outcome):
            raise ValueError(f"task outcome fence lost for attempt {attempt_id}")

    def fail(self, request: FinalizeTaskFailure) -> FailureDisposition:
        disposition = finalize_failure(self._graph, request)
        if disposition.status in {"unknown_start", "lost_fence"}:
            _logger.warning(
                "task failure not recorded as attempt: node_id=%d run_id=%s attempt_id=%s "
                + "status=%s reason=%s",
                request.node_id,
                request.run_id,
                request.attempt_id,
                disposition.status,
                disposition.reason,
            )
        return disposition
