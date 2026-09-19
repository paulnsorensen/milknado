"""Owner and observer command capability builders."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from milknado.domains.graph.commands import OwnerCapabilities
from milknado.web.commands import (
    GitInspection,
    GraphEditCommands,
    GraphProtocol,
    ReviewHandler,
    RunHandler,
    SchedulingHandler,
    SessionInputHandler,
    WebCommands,
)


@dataclass(frozen=True, slots=True)
class HostDependencies:
    graph: GraphProtocol | None = None
    flavor_registry: frozenset[str] = frozenset()
    project_root: Path = Path()
    review_decision: ReviewHandler | None = None
    git: GitInspection | None = None
    owner_capabilities: OwnerCapabilities | None = None


@dataclass(frozen=True, slots=True)
class OwnerHandlers:
    session_input: SessionInputHandler | None = None
    cancel: RunHandler | None = None
    force_stop: RunHandler | None = None
    stop_scheduling: SchedulingHandler | None = None


@dataclass(frozen=True, slots=True)
class ObserverHandlers:
    session_input: SessionInputHandler | None = None
    cancel: RunHandler | None = None


def _graph_edits(dependencies: HostDependencies) -> GraphEditCommands | None:
    if dependencies.graph is None:
        return None
    return GraphEditCommands(
        dependencies.graph, dependencies.flavor_registry, dependencies.project_root
    )


def owner_commands(
    handlers: OwnerHandlers | None = None,
    dependencies: HostDependencies | None = None,
) -> WebCommands:
    handlers = handlers or OwnerHandlers()
    dependencies = dependencies or HostDependencies()
    return WebCommands(
        session_input=handlers.session_input,
        cancel=handlers.cancel,
        force_stop=handlers.force_stop,
        stop_scheduling=handlers.stop_scheduling,
        graph_edits=_graph_edits(dependencies),
        review_decision=dependencies.review_decision,
        git=dependencies.git,
        owner_capabilities=dependencies.owner_capabilities,
    )


def observer_commands(
    handlers: ObserverHandlers | None = None,
    dependencies: HostDependencies | None = None,
) -> WebCommands:
    handlers = handlers or ObserverHandlers()
    dependencies = dependencies or HostDependencies()
    return WebCommands(
        session_input=handlers.session_input,
        cancel=handlers.cancel,
        graph_edits=_graph_edits(dependencies),
        review_decision=dependencies.review_decision,
        git=dependencies.git,
    )
