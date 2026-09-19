"""Command capabilities exposed by the web adapter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from milknado.domains.common import MikadoNode, NodeKind, NodeSpec, SessionInput
from milknado.domains.graph.commands import (
    CommandReceipt,
    OwnerCapabilities,
)
from milknado.domains.graph.goal_review import GoalReviewRecord


class SessionInputHandler(Protocol):
    def __call__(self, request: SessionInput) -> CommandReceipt: ...


class RunHandler(Protocol):
    def __call__(self, run_id: str) -> dict[str, object]: ...


class SchedulingHandler(Protocol):
    def __call__(self) -> None: ...


class ReviewHandler(Protocol):
    def __call__(self, review_id: int, decision: str) -> GoalReviewRecord: ...


class GraphProtocol(Protocol):
    def add_node(
        self,
        description: str,
        parent_id: int | None = None,
        spec: NodeSpec | None = None,
        files: tuple[str, ...] | None = None,
    ) -> MikadoNode: ...

    def update_node(  # noqa: PLR0913 - mirrors graph mutation contract
        self,
        node_id: int,
        description: str | None = None,
        kind: NodeKind | None = None,
        flavor: str | None = None,
        artifact_path: str | None = None,
        flavor_registry: frozenset[str] | None = None,
    ) -> None: ...

    def move_node(self, node_id: int, parent_id: int | None) -> None: ...
    def archive_subtree(self, node_id: int) -> int: ...


@dataclass(frozen=True, slots=True)
class GraphEditCommands:
    graph: GraphProtocol
    flavor_registry: frozenset[str]
    project_root: Path


class GitInspection(Protocol):
    def changes(self, run_id: str) -> list[dict[str, object]]: ...

    def diff(self, run_id: str, path: str) -> str: ...


@dataclass(frozen=True, slots=True)
class WebCommands:
    session_input: SessionInputHandler | None = None
    cancel: RunHandler | None = None
    force_stop: RunHandler | None = None
    stop_scheduling: SchedulingHandler | None = None
    graph_edits: GraphEditCommands | None = None
    review_decision: ReviewHandler | None = None
    git: GitInspection | None = None
    owner_capabilities: OwnerCapabilities | None = None


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


def _owner_capabilities(owner: OwnerCapabilities | None) -> dict[str, object]:
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
