"""Command capabilities exposed by the web adapter."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from milknado.adapters import ChangedFile
from milknado.domains.common import SessionContext, SessionInput
from milknado.domains.graph import (
    GoalReviewDecisionRequest,
    GoalReviewRecord,
    MikadoGraph,
    OwnerCapabilities,
)

OwnerCapabilitiesProvider = Callable[[str | None], OwnerCapabilities | None]


class SessionInputHandler(Protocol):
    def __call__(self, run_id: str, request: SessionInput) -> SessionInput | None: ...


class RunHandler(Protocol):
    def __call__(self, run_id: str) -> dict[str, object]: ...


class SchedulingHandler(Protocol):
    def __call__(self) -> None: ...


class ReviewHandler(Protocol):
    def __call__(
        self, request: GoalReviewDecisionRequest, *, decided_by: str
    ) -> GoalReviewRecord: ...


@dataclass(frozen=True, slots=True)
class GraphEditCommands:
    graph: MikadoGraph
    flavor_registry: frozenset[str]
    project_root: Path


class GitInspection(Protocol):
    def changes(self, context: SessionContext) -> tuple[ChangedFile, ...]: ...
    def diff(self, context: SessionContext, path: str) -> str: ...


@dataclass(frozen=True, slots=True)
class WebCommands:
    session_input: SessionInputHandler | None = None
    cancel: RunHandler | None = None
    force_stop: RunHandler | None = None
    stop_scheduling: SchedulingHandler | None = None
    graph_edits: GraphEditCommands | None = None
    review_decision: ReviewHandler | None = None
    git: GitInspection | None = None
    owner_capabilities: OwnerCapabilities | OwnerCapabilitiesProvider | None = None


def _capability(value: object | None, reason: str) -> dict[str, object]:
    return {"available": value is not None, "reason": None if value is not None else reason}


def build_capabilities(commands: WebCommands) -> dict[str, object]:
    """Return the web-owned capability projection."""
    return {
        "session_input": _capability(commands.session_input, "Session input is unavailable."),
        "cancel": _capability(commands.cancel, "Cancel is unavailable."),
        "force_stop": _capability(commands.force_stop, "Force stop is unavailable."),
        "stop_scheduling": _capability(
            commands.stop_scheduling, "Stop scheduling is unavailable."
        ),
        "graph_edits": _capability(commands.graph_edits, "Graph edits are unavailable."),
        "review_decision": _capability(
            commands.review_decision, "Review decisions are unavailable."
        ),
        "git": _capability(commands.git, "Git inspection is unavailable."),
        "owner": _owner_capabilities(commands.owner_capabilities),
    }


def _owner_capabilities(
    owner: OwnerCapabilities | OwnerCapabilitiesProvider | None,
    run_id: str | None = None,
) -> dict[str, object]:
    if callable(owner):
        owner = owner(run_id)
    if owner is None:
        return {"available": False, "reason": "No live owner is connected."}
    return {
        "available": True,
        "run_id": owner.run_id,
        "node_id": owner.node_id,
        "invocation_id": owner.invocation_id,
        "owner_incarnation": owner.owner_incarnation,
        "actions": owner.actions,
        "permission_ids": owner.permission_ids,
        "published_at": owner.published_at,
    }
