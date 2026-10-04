from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import msgspec

from milknado.domains.coordinator.commands import (
    CoordinatorAction,
    DispatchHandoff,
    submit_coordinator_action,
)
from milknado.domains.coordinator.control_models import (
    AttemptCommand,
    CoordinatorCommand,
    CoordinatorCommandReceipt,
    CreateGroup,
    DecideGoalReview,
    DispatchTask,
    FailLaunch,
    FinishTask,
    PlanGoal,
    RecordRevision,
    Recover,
    RequestGoalReview,
    RuntimeAction,
    StartGoal,
)
from milknado.domains.coordinator.persistence import get_coordinator
from milknado.domains.coordinator.projection import (
    CoordinatorSnapshot,
    read_coordinator_snapshot,
)
from milknado.domains.coordinator.recovery import RecoveryRuntime, recover_coordinator
from milknado.domains.coordinator.workflow import CoordinatorWorkflow
from milknado.domains.execution import NodeLoopOutcome
from milknado.domains.graph import (
    GoalReviewDecisionRequest,
    GoalReviewRequest,
    GroupWorkspace,
    MikadoGraph,
    TaskAttempt,
)
from milknado.domains.planning import Planner
from milknado.loop.sessions import RuntimeSession


@dataclass(frozen=True, slots=True)
class CoordinatorServices:
    planner: Planner | None = None
    runtime_session: Callable[[str], RuntimeSession | None] | None = None
    recovery_runtime: RecoveryRuntime | None = None


class CoordinatorControl:
    def __init__(
        self, graph: MikadoGraph, project_root: Path, services: CoordinatorServices | None = None
    ) -> None:
        self._graph: MikadoGraph = graph
        self._root: Path = project_root
        self._services: CoordinatorServices = services or CoordinatorServices()

    @property
    def _conn(self) -> sqlite3.Connection:
        return self._graph.group_connection

    def read_coordinator_snapshot(self, session_id: str, cursor: int) -> CoordinatorSnapshot:
        return read_coordinator_snapshot(self._graph, self._conn, session_id, cursor)

    def send_coordinator_command(
        self, session_id: str, command: CoordinatorCommand
    ) -> CoordinatorCommandReceipt:
        if not command.command_id:
            raise ValueError("command_id must not be empty")
        if isinstance(command, StartGoal) != (not session_id):
            raise ValueError(
                "start_goal requires an empty session ID; other commands require a session ID"
            )
        with self._graph.synchronization_lock:
            if session_id and get_coordinator(self._conn, session_id) is None:
                raise KeyError(session_id)
            fingerprint = hashlib.sha256(msgspec.json.encode(command)).hexdigest()
            existing = self._reserve(session_id, command.command_id, fingerprint)
            if existing is not None:
                return existing
            try:
                status, result = self._execute(session_id, command)
            except ValueError as error:
                status, result = "rejected", str(error)
            return self._complete(session_id, command.command_id, status, result)

    def _reserve(
        self, session_id: str, command_id: str, fingerprint: str
    ) -> CoordinatorCommandReceipt | None:
        with self._conn:
            _ = self._conn.execute("""
                CREATE TABLE IF NOT EXISTS coordinator_web_receipts (
                    command_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    command_hash TEXT NOT NULL,
                    status TEXT NOT NULL,
                    result_json TEXT NOT NULL
                )
            """)
            cursor = self._conn.execute(
                "INSERT OR IGNORE INTO coordinator_web_receipts "
                + "(command_id, session_id, command_hash, status, result_json) "
                + "VALUES (?, ?, ?, 'unconfirmed', 'null')",
                (command_id, session_id, fingerprint),
            )
        row = cast(
            tuple[str, str, str, str] | None,
            self._conn.execute(
                "SELECT session_id, command_hash, status, result_json "
                + "FROM coordinator_web_receipts WHERE command_id = ?",
                (command_id,),
            ).fetchone(),
        )
        if row is None:
            raise RuntimeError("coordinator command has no receipt")
        if row[:2] != (session_id, fingerprint):
            raise ValueError("command_id was reused for a different command")
        if cursor.rowcount:
            return None
        return CoordinatorCommandReceipt(
            command_id,
            session_id,
            cast(
                Literal["accepted", "unavailable", "unsupported", "rejected", "unconfirmed"],
                row[2],
            ),
            cast(object, msgspec.json.decode(row[3].encode())),
        )

    def _complete(
        self,
        session_id: str,
        command_id: str,
        status: Literal["accepted", "unavailable", "unsupported", "rejected"],
        result: object,
    ) -> CoordinatorCommandReceipt:
        built = cast(object, msgspec.to_builtins(result))
        with self._conn:
            _ = self._conn.execute(
                "UPDATE coordinator_web_receipts SET status = ?, result_json = ? "
                + "WHERE command_id = ?",
                (status, msgspec.json.encode(built).decode(), command_id),
            )
        return CoordinatorCommandReceipt(command_id, session_id, status, built)

    def _execute(
        self, session_id: str, command: CoordinatorCommand
    ) -> tuple[Literal["accepted", "unavailable", "unsupported"], object]:
        workflow = CoordinatorWorkflow(self._graph, self._conn)
        if isinstance(command, StartGoal):
            return "accepted", workflow.start_goal(command.description, command.provider)
        session = get_coordinator(self._conn, session_id)
        assert session is not None
        match command:
            case PlanGoal():
                if self._services.planner is None:
                    return "unavailable", "Planner is not connected."
                return "accepted", workflow.plan_goal(
                    session, self._services.planner, self._root, command.command_id
                )
            case CreateGroup():
                workspace = GroupWorkspace(
                    command.worktree_path, command.branch_name, command.provider_session_id
                )
                return "accepted", workflow.create_group(
                    session, command.graph_id, command.tasks, workspace
                )
            case DispatchTask():
                return "accepted", workflow.dispatch_task(
                    session, command.group_id, command.node_id, command.run_id
                )
            case AttemptCommand() | FailLaunch() | FinishTask():
                attempt = TaskAttempt(
                    command.group_id, command.node_id, command.run_id, command.attempt_id
                )
                if isinstance(command, AttemptCommand):
                    return "accepted", workflow.acknowledge_launch(
                        session, DispatchHandoff(attempt, "awaiting_launch")
                    )
                if isinstance(command, FailLaunch):
                    workflow.fail_launch(
                        session, DispatchHandoff(attempt, "awaiting_launch"), command.reason
                    )
                    return "accepted", None
                workflow.finish_task(
                    session,
                    attempt,
                    NodeLoopOutcome(
                        command.node_id,
                        command.success,
                        command.detail,
                        ownership_preserved=command.ownership_preserved,
                    ),
                )
                return "accepted", None
            case RecordRevision():
                workflow.record_revision(session, command.revision_id, command.affected_node_ids)
                return "accepted", None
            case RequestGoalReview():
                request = GoalReviewRequest(
                    session.goal_id,
                    command.goal_revision,
                    command.evidence,
                    command.proposed_change,
                    command.affected_node_ids,
                    operation_id=command.command_id,
                )
                return "accepted", workflow.review_goal_change(session, request)
            case DecideGoalReview():
                linked = cast(
                    tuple[int] | None,
                    self._conn.execute(
                        "SELECT 1 FROM coordinator_links WHERE session_id = ? "
                        + "AND kind = 'approval' AND entity_id = ?",
                        (session_id, str(command.review_id)),
                    ).fetchone(),
                )
                if linked is None:
                    raise ValueError("review belongs to another coordinator")
                return "accepted", self._graph.decide_goal_review(
                    GoalReviewDecisionRequest(command.review_id, command.decision),
                    decided_by=command.decided_by,
                )
            case RuntimeAction():
                if self._services.runtime_session is None:
                    return "unavailable", "Provider runtime is not connected."
                runtime = self._services.runtime_session(command.provider_session_id)
                if runtime is None:
                    return "unavailable", "Provider session is not active."
                return "accepted", submit_coordinator_action(
                    self._conn,
                    session,
                    runtime,
                    CoordinatorAction(command.command_id, command.input),
                )
            case Recover():
                if self._services.recovery_runtime is None:
                    return "unavailable", "Recovery runtime is not connected."
                return "accepted", recover_coordinator(
                    self._conn, session_id, self._services.recovery_runtime
                )
