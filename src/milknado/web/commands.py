# pyright: reportAny=false, reportExplicitAny=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnannotatedClassAttribute=false, reportUnnecessaryCast=false, reportUnnecessaryIsInstance=false
"""Command capabilities exposed by the web adapter."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from milknado.domains.graph.commands import OwnerCapabilities

CommandFn = Callable[..., object]


@dataclass(frozen=True, slots=True)
class WebCommands:
    session_input: CommandFn | None = None
    cancel: CommandFn | None = None
    force_stop: CommandFn | None = None
    stop_scheduling: CommandFn | None = None
    graph_edits: object | None = None
    review_decision: CommandFn | None = None
    git: object | None = None
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
