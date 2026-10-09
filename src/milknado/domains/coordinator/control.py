from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import msgspec

import milknado.domains.coordinator.turns as turns
from milknado.domains.coordinator._command_lifecycle import CommandLifecycle
from milknado.domains.coordinator.attempt_commands import apply_attempt_command
from milknado.domains.coordinator.control_models import (
    AttemptCommand,
    CancelTurn,
    CoordinatorCommand,
    CoordinatorCommandReceipt,
    CreateGroup,
    DecideGoalReview,
    DecidePlanProposal,
    DispatchTask,
    FailLaunch,
    FinishTask,
    PlanGoal,
    RecordRevision,
    Recover,
    RequestGoalReview,
    RuntimeAction,
    StartGoal,
    StartTurn,
)
from milknado.domains.coordinator.control_services import (
    CoordinatorServices,
    TurnPreflightError,
    TurnRuntimePort,
    TurnRuntimeRequest,
)
from milknado.domains.coordinator.model import CoordinatorSession, CoordinatorSessionSummary
from milknado.domains.coordinator.persistence import (
    get_coordinator,
    list_coordinator_sessions,
)
from milknado.domains.coordinator.projection import (
    CoordinatorSnapshot,
    read_coordinator_snapshot,
)
from milknado.domains.coordinator.receipt_results import reserve_command_receipt
from milknado.domains.coordinator.recovery import recover_coordinator
from milknado.domains.coordinator.review_decisions import (
    decide_coordinator_review,
    decide_goal_review,
    request_coordinator_review,
)
from milknado.domains.coordinator.runtime_actions import ActionInvocation, send_runtime_action
from milknado.domains.coordinator.turn_context import TurnContext
from milknado.domains.coordinator.turn_fences import (
    cancel_owned_turn,
    claim_turn,
    reconcile_turn_fences,
    release_unconfirmed_turn,
)
from milknado.domains.coordinator.workflow import CoordinatorWorkflow
from milknado.domains.graph import (
    ControllerAuthorizationError,
    GoalReviewDecisionRequest,
    GoalReviewRecord,
    GroupWorkspace,
    MikadoGraph,
)


class CoordinatorControl:
    def __init__(
        self, graph: MikadoGraph, project_root: Path, services: CoordinatorServices | None = None
    ) -> None:
        self._graph: MikadoGraph = graph
        self._root: Path = project_root
        self._services: CoordinatorServices = services or CoordinatorServices()
        self._commands: CommandLifecycle = CommandLifecycle(lambda: self._conn)

    @property
    def _conn(self) -> sqlite3.Connection:
        return self._graph.group_connection

    def read_coordinator_snapshot(self, session_id: str, cursor: int) -> CoordinatorSnapshot:
        return read_coordinator_snapshot(self._graph, self._conn, session_id, cursor)

    def list_coordinator_sessions(self) -> tuple[CoordinatorSessionSummary, ...]:
        with self._graph.synchronization_lock:
            return list_coordinator_sessions(self._conn)

    def decide_goal_review(
        self, request: GoalReviewDecisionRequest, *, decided_by: str
    ) -> GoalReviewRecord:
        with self._commands.command():
            return decide_goal_review(self._graph, self._services, request, decided_by)

    def send_coordinator_command(
        self, session_id: str, command: CoordinatorCommand
    ) -> CoordinatorCommandReceipt:
        with self._commands.command():
            return self._send_command(session_id, command)

    def shutdown(self, stop_runtime: Callable[[], None]) -> None:
        self._commands.shutdown(stop_runtime)

    def _send_command(
        self, session_id: str, command: CoordinatorCommand
    ) -> CoordinatorCommandReceipt:
        if not command.command_id:
            raise ValueError("command_id must not be empty")
        if isinstance(command, StartGoal) != (not session_id):
            raise ValueError(
                "start_goal requires an empty session ID; other commands require a session ID"
            )
        if isinstance(command, StartTurn):
            return self._send_turn(session_id, command)
        if isinstance(command, RuntimeAction):
            return send_runtime_action(
                self._graph,
                self._services,
                ActionInvocation(session_id, command),
                self._commands.complete,
            )
        if isinstance(command, Recover) and self._services.recovery_runtime is not None:
            workers = self._services.recovery_runtime.workers
            if workers is not None:
                reconcile_turn_fences(self._conn, self._graph, session_id, workers)
        with self._graph.synchronization_lock:
            if session_id and get_coordinator(self._conn, session_id) is None:
                raise KeyError(session_id)
            fingerprint = hashlib.sha256(msgspec.json.encode(command)).hexdigest()
            existing = reserve_command_receipt(
                self._conn, session_id, command.command_id, fingerprint
            )
            if existing is not None:
                return existing
            if not isinstance(command, (PlanGoal, DecidePlanProposal, Recover)):
                return self._execute_reserved(session_id, command)
        return self._execute_reserved(session_id, command)

    def _execute_reserved(
        self, session_id: str, command: CoordinatorCommand
    ) -> CoordinatorCommandReceipt:
        try:
            status, result = self._execute(session_id, command)
        except (ValueError, PermissionError, ControllerAuthorizationError) as error:
            status, result = "rejected", str(error)
        with self._graph.synchronization_lock:
            return self._commands.complete(session_id, command, status, result)

    def _admit_turn(
        self, context: TurnContext, command: StartTurn
    ) -> CoordinatorCommandReceipt | tuple[turns.TurnLaunch, TurnRuntimePort]:
        session = get_coordinator(context.conn, context.session_id)
        if session is None:
            raise KeyError(context.session_id)
        fingerprint = hashlib.sha256(msgspec.json.encode(command)).hexdigest()
        existing = reserve_command_receipt(
            context.conn, context.session_id, context.command_id, fingerprint
        )
        if existing is not None:
            return existing
        runtime = self._services.turn_runtime
        if runtime is None:
            return self._commands.complete(
                context.session_id, command, "unavailable", "Turn runtime is not connected."
            )
        try:
            launch = turns.prepare_turn(context.conn, self._graph, session, command)
            turn_owner = self._services.turn_owner
            owner = turn_owner(context.command_id) if turn_owner else None
            claim_turn(context, launch, owner)
        except (ValueError, PermissionError) as error:
            return self._commands.complete(context.session_id, command, "rejected", str(error))
        return launch, runtime

    def _send_turn(self, session_id: str, command: StartTurn) -> CoordinatorCommandReceipt:
        context = TurnContext(self._conn, session_id, command.command_id)
        with self._graph.synchronization_lock:
            admission = self._admit_turn(context, command)
            if isinstance(admission, CoordinatorCommandReceipt):
                return admission
        launch, runtime_port = admission
        hooks = turns.make_turn_hooks(context, self._graph, launch)
        request = TurnRuntimeRequest(
            launch.provider, command.prompt, launch.group, launch.identity, hooks, launch.attempt
        )
        try:
            runtime = runtime_port.run(request)
        except TurnPreflightError as error:
            with self._graph.synchronization_lock:
                release_unconfirmed_turn(self._conn, command.command_id)
                return self._commands.complete(session_id, command, "unavailable", str(error))
        except (OSError, ValueError) as error:
            with self._graph.synchronization_lock:
                return self._commands.complete(session_id, command, "unavailable", str(error))
        with self._graph.synchronization_lock:
            try:
                status, result = turns.finish_turn(context, launch, runtime, self._root)
            except (ValueError, sqlite3.IntegrityError) as error:
                status, result = "rejected", str(error)
            release_unconfirmed_turn(self._conn, command.command_id)
            return self._commands.complete(session_id, command, status, result)

    def _execute(
        self, session_id: str, command: CoordinatorCommand
    ) -> tuple[Literal["accepted", "unavailable", "unsupported"], object]:
        workflow = CoordinatorWorkflow(self._graph, self._conn)
        if isinstance(command, StartGoal):
            return "accepted", workflow.start_goal(command.description, command.provider)
        with self._graph.synchronization_lock:
            session = get_coordinator(self._conn, session_id)
            assert session is not None
        match command:
            case PlanGoal() | CreateGroup() | DispatchTask() | RecordRevision():
                return self._workflow_command(workflow, session, command)
            case DecidePlanProposal():
                if self._services.planner is None:
                    return "unavailable", "Planner is not connected."
                return "accepted", workflow.decide_plan(
                    session,
                    self._services.planner,
                    self._root,
                    command.proposal_id,
                    command.decision,
                )
            case AttemptCommand() | FailLaunch() | FinishTask():
                return "accepted", apply_attempt_command(workflow, session, command)
            case RequestGoalReview():
                return "accepted", request_coordinator_review(workflow, session, command)
            case DecideGoalReview():
                return decide_coordinator_review(self._graph, self._services, session, command)
            case CancelTurn():
                return cancel_owned_turn(
                    self._conn, session_id, command.turn_id, self._services.turn_cancel
                )
            case Recover():
                return self._recover(session)
            case StartTurn() | RuntimeAction():
                raise AssertionError("runtime command must run outside the graph lock")

    def _recover(
        self, session: CoordinatorSession
    ) -> tuple[Literal["accepted", "unavailable"], object]:
        runtime = self._services.recovery_runtime
        if runtime is None:
            return "unavailable", "Recovery runtime is not connected."
        return "accepted", recover_coordinator(
            self._conn,
            session.id,
            runtime,
            synchronization_lock=self._graph.synchronization_lock,
        )

    def _workflow_command(
        self,
        workflow: CoordinatorWorkflow,
        session: CoordinatorSession,
        command: PlanGoal | CreateGroup | DispatchTask | RecordRevision,
    ) -> tuple[Literal["accepted", "unavailable"], object]:
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
            case RecordRevision():
                workflow.record_revision(session, command.revision_id, command.affected_node_ids)
                return "accepted", None
