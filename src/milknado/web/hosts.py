"""Owner and observer command capability builders."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

from milknado.domains.common import SessionInput
from milknado.domains.graph.commands import CommandReceipt, OwnerCapabilities
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


class OwnerController(Protocol):
    def session_input(self, run_id: str, command: SessionInput) -> bool: ...
    def cancel(self, run_id: str) -> None: ...
    def force_stop(self, run_id: str, timeout: float = 10.0) -> bool: ...
    def stop_scheduling(self) -> None: ...


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
    handlers: OwnerHandlers | OwnerController | None = None,
    dependencies: HostDependencies | None = None,
) -> WebCommands:
    dependencies = dependencies or HostDependencies()
    if handlers is None:
        handlers = OwnerHandlers()
    elif not isinstance(handlers, OwnerHandlers):
        controller = handlers
        handlers = OwnerHandlers(
            session_input=lambda request: cast(
                CommandReceipt,
                cast(
                    object,
                    controller.session_input(
                        dependencies.owner_capabilities.run_id
                        if dependencies.owner_capabilities is not None
                        else request.request_id,
                        request,
                    ),
                ),
            ),
            cancel=lambda run_id: {"run_id": run_id, "result": controller.cancel(run_id)},
            force_stop=lambda run_id: {
                "run_id": run_id,
                "result": controller.force_stop(run_id),
            },
            stop_scheduling=controller.stop_scheduling,
        )
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
