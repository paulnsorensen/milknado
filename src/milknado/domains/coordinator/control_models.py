from __future__ import annotations

from typing import Annotated, Literal

import msgspec

from milknado.domains.common import SessionInput
from milknado.domains.graph import GoalReviewDecision

_SQLiteIdentifier = Annotated[int, msgspec.Meta(ge=-(2**63), le=2**63 - 1)]


class StartGoal(
    msgspec.Struct, frozen=True, tag="start_goal", tag_field="kind", forbid_unknown_fields=True
):
    command_id: str
    description: str
    provider: Literal["claude", "codex"]


class PlanGoal(
    msgspec.Struct, frozen=True, tag="plan_goal", tag_field="kind", forbid_unknown_fields=True
):
    command_id: str


class DecidePlanProposal(
    msgspec.Struct,
    frozen=True,
    tag="decide_plan_proposal",
    tag_field="kind",
    forbid_unknown_fields=True,
):
    command_id: str
    proposal_id: str
    decision: Literal["accepted", "rejected"]


class CreateGroup(
    msgspec.Struct, frozen=True, tag="create_group", tag_field="kind", forbid_unknown_fields=True
):
    command_id: str
    graph_id: str
    tasks: tuple[_SQLiteIdentifier, ...]
    worktree_path: str
    branch_name: str
    provider_session_id: str | None = None


class StartTurn(
    msgspec.Struct, frozen=True, tag="start_turn", tag_field="kind", forbid_unknown_fields=True
):
    command_id: str
    prompt: str
    provider: Literal["claude", "codex"] | None = None
    group_id: str | None = None
    node_id: _SQLiteIdentifier | None = None
    run_id: str | None = None
    attempt_id: str | None = None


class CancelTurn(
    msgspec.Struct, frozen=True, tag="cancel_turn", tag_field="kind", forbid_unknown_fields=True
):
    command_id: str
    turn_id: str


class DispatchTask(
    msgspec.Struct, frozen=True, tag="dispatch_task", tag_field="kind", forbid_unknown_fields=True
):
    command_id: str
    group_id: str
    node_id: int
    run_id: str


class AttemptCommand(
    msgspec.Struct,
    frozen=True,
    tag="acknowledge_launch",
    tag_field="kind",
    forbid_unknown_fields=True,
):
    command_id: str
    group_id: str
    node_id: int
    run_id: str
    attempt_id: str


class FailLaunch(
    msgspec.Struct, frozen=True, tag="fail_launch", tag_field="kind", forbid_unknown_fields=True
):
    command_id: str
    group_id: str
    node_id: int
    run_id: str
    attempt_id: str
    reason: str


class FinishTask(
    msgspec.Struct, frozen=True, tag="finish_task", tag_field="kind", forbid_unknown_fields=True
):
    command_id: str
    group_id: str
    node_id: int
    run_id: str
    attempt_id: str
    success: bool
    detail: str = ""
    ownership_preserved: bool = False


class RecordRevision(
    msgspec.Struct,
    frozen=True,
    tag="record_revision",
    tag_field="kind",
    forbid_unknown_fields=True,
):
    command_id: str
    revision_id: str
    affected_node_ids: tuple[_SQLiteIdentifier, ...]


class RequestGoalReview(
    msgspec.Struct,
    frozen=True,
    tag="request_goal_review",
    tag_field="kind",
    forbid_unknown_fields=True,
):
    command_id: str
    goal_revision: str
    evidence: str
    proposed_change: str
    reviewer: str
    affected_node_ids: tuple[int, ...] | None = None


class DecideGoalReview(
    msgspec.Struct,
    frozen=True,
    tag="decide_goal_review",
    tag_field="kind",
    forbid_unknown_fields=True,
):
    command_id: str
    review_id: int
    decision: GoalReviewDecision
    decided_by: str = "web"


class RuntimeAction(
    msgspec.Struct, frozen=True, tag="runtime_action", tag_field="kind", forbid_unknown_fields=True
):
    command_id: str
    provider_session_id: str
    input: SessionInput


class Recover(
    msgspec.Struct, frozen=True, tag="recover", tag_field="kind", forbid_unknown_fields=True
):
    command_id: str


CoordinatorCommand = (
    StartGoal
    | PlanGoal
    | DecidePlanProposal
    | CreateGroup
    | StartTurn
    | CancelTurn
    | DispatchTask
    | AttemptCommand
    | FailLaunch
    | FinishTask
    | RecordRevision
    | RequestGoalReview
    | DecideGoalReview
    | RuntimeAction
    | Recover
)


class CoordinatorCommandReceipt(msgspec.Struct, frozen=True):
    command_id: str
    session_id: str
    status: Literal["accepted", "unavailable", "unsupported", "rejected", "unconfirmed"]
    result: object
