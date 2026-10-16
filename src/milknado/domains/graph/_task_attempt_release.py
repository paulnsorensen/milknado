"""Attempt-lifecycle writes that bracket the release of a gated worker."""

from __future__ import annotations

from dataclasses import dataclass

from milknado.domains.graph._reads import get_node
from milknado.domains.graph._run_persistence import get_run
from milknado.domains.graph._task_attempt_binding import bind_worker_invocation
from milknado.domains.graph._task_attempt_reads import get_attempt
from milknado.domains.graph._task_attempts import AttemptGraphPort


@dataclass(frozen=True, slots=True)
class ReleaseFence:
    attempt_id: str
    owner_run_id: str
    invocation_id: str
    budgeted: bool

    @property
    def ids(self) -> tuple[str, str, str]:
        return (self.attempt_id, self.owner_run_id, self.invocation_id)


def begin_release(graph: AttemptGraphPort, run_id: str, invocation_id: str) -> ReleaseFence | None:
    """Bind the worker invocation to the run's attempt; None when the run has no attempt."""
    with graph.synchronization_lock:
        conn = graph.group_connection
        row = get_run(conn, run_id)
        if row is None:
            raise RuntimeError("graph run missing before worker release")
        attempt_id = row["attempt_id"]
        if attempt_id is None:
            return None
        attempt = get_attempt(conn, attempt_id)
        if attempt is None:
            raise RuntimeError("linked task attempt missing before worker release")
        node = get_node(conn, row["node_id"])
        if node is None or node.run_id is None:
            raise RuntimeError("task attempt owner fence lost before worker release")
        if not bind_worker_invocation(conn, (attempt_id, node.run_id, run_id, invocation_id)):
            raise RuntimeError("task attempt invocation fence lost before worker release")
        return ReleaseFence(attempt_id, node.run_id, invocation_id, attempt.max_node_attempts > 1)
