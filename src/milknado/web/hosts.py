"""Owner and observer command capability builders."""

from __future__ import annotations

from pathlib import Path

from milknado.domains.graph.commands import OwnerCapabilities
from milknado.web.commands import (
    GitInspection,
    GraphEditCommands,
    ReviewHandler,
    RunHandler,
    SchedulingHandler,
    SessionInputHandler,
    WebCommands,
)


def owner_commands(  # noqa: PLR0913 - approved capability set
    session_input: SessionInputHandler | None = None,
    cancel: RunHandler | None = None,
    force_stop: RunHandler | None = None,
    stop_scheduling: SchedulingHandler | None = None,
    *,
    graph: object | None = None,
    flavor_registry: frozenset[str] | None = None,
    project_root: Path | None = None,
    review_decision: ReviewHandler | None = None,
    git: GitInspection | None = None,
    owner_capabilities: OwnerCapabilities | None = None,
) -> WebCommands:
    graph_edits = (
        None
        if graph is None
        else GraphEditCommands(graph, flavor_registry or frozenset(), project_root or Path())
    )
    return WebCommands(
        session_input,
        cancel,
        force_stop,
        stop_scheduling,
        graph_edits,
        review_decision,
        git,
        owner_capabilities,
    )


def observer_commands(  # noqa: PLR0913 - approved capability set
    session_input: SessionInputHandler | None = None,
    cancel: RunHandler | None = None,
    *,
    graph: object | None = None,
    flavor_registry: frozenset[str] | None = None,
    project_root: Path | None = None,
    review_decision: ReviewHandler | None = None,
    git: GitInspection | None = None,
) -> WebCommands:
    graph_edits = (
        None
        if graph is None
        else GraphEditCommands(graph, flavor_registry or frozenset(), project_root or Path())
    )
    return WebCommands(
        session_input=session_input,
        cancel=cancel,
        graph_edits=graph_edits,
        review_decision=review_decision,
        git=git,
    )
