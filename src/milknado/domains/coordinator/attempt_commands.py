"""Apply a coordinator-owned task attempt command."""

from __future__ import annotations

from milknado.domains.coordinator.commands import DispatchHandoff
from milknado.domains.coordinator.control_models import AttemptCommand, FailLaunch, FinishTask
from milknado.domains.coordinator.model import CoordinatorSession
from milknado.domains.coordinator.workflow import CoordinatorWorkflow
from milknado.domains.execution import NodeLoopOutcome
from milknado.domains.graph import TaskAttempt


def apply_attempt_command(
    workflow: CoordinatorWorkflow,
    session: CoordinatorSession,
    command: AttemptCommand | FailLaunch | FinishTask,
) -> object:
    attempt = TaskAttempt(command.group_id, command.node_id, command.run_id, command.attempt_id)
    if isinstance(command, AttemptCommand):
        return workflow.acknowledge_launch(session, DispatchHandoff(attempt, "awaiting_launch"))
    if isinstance(command, FailLaunch):
        workflow.fail_launch(session, DispatchHandoff(attempt, "awaiting_launch"), command.reason)
        return None
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
    return None
