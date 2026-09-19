"""Owner and observer command capability builders."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

from milknado.domains.common import GitPort, SessionInput
from milknado.domains.dispatch import ProcessTerminationPort, cancel_run
from milknado.domains.graph import MikadoGraph, OwnerCapabilities, admit_session_command
from milknado.web.commands import (
    GitInspection,
    GraphEditCommands,
    OwnerCapabilitiesProvider,
    ReviewHandler,
    RunHandler,
    SchedulingHandler,
    SessionInputHandler,
    WebCommands,
)


class OwnerController(Protocol):
    def session_input(self, run_id: str, command: SessionInput) -> bool: ...
    def cancel(self, run_id: str) -> dict[str, object] | None: ...
    def force_stop(self, run_id: str, timeout: float = 10.0) -> bool: ...
    def stop_scheduling(self) -> None: ...


@dataclass(frozen=True, slots=True)
class HostDependencies:
    graph: MikadoGraph | None = None
    flavor_registry: frozenset[str] = frozenset()
    project_root: Path = Path()
    git_port: GitPort | None = None
    process: ProcessTerminationPort | None = None
    review_decision: ReviewHandler | None = None
    git: GitInspection | None = None
    owner_capabilities: OwnerCapabilities | OwnerCapabilitiesProvider | None = None


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


def _current_owner_capabilities(
    dependencies: HostDependencies,
    run_id: str | None = None,
) -> OwnerCapabilities | None:
    owner = dependencies.owner_capabilities
    return owner(run_id) if callable(owner) else owner


def _cancel_owner(
    controller: OwnerController, graph: MikadoGraph | None, run_id: str
) -> dict[str, object]:
    result = controller.cancel(run_id)
    if graph is None:
        if isinstance(result, dict):
            return result
        raise RuntimeError("owner cancellation requires an injected graph")
    record = graph.runs.get(run_id)
    if record is None:
        raise ValueError(f"run {run_id!r} not found after cancellation")
    return cast(dict[str, object], cast(object, record))


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
            session_input=lambda run_id, request: (
                request
                if (
                    (
                        (owner := _current_owner_capabilities(dependencies, run_id)) is None
                        or run_id == owner.run_id
                    )
                    and controller.session_input(run_id, request)
                )
                else None
            ),
            cancel=lambda run_id: _cancel_owner(controller, dependencies.graph, run_id),
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
    if handlers.session_input is None and dependencies.graph is not None:
        graph = dependencies.graph
        handlers = ObserverHandlers(
            session_input=lambda run_id, request: admit_session_command(
                graph,
                run_id,
                request,
                owner_incarnation=(
                    owner.owner_incarnation
                    if (owner := _current_owner_capabilities(dependencies, run_id)) is not None
                    else None
                ),
            ),
            cancel=handlers.cancel,
        )
    if (
        handlers.cancel is None
        and dependencies.graph is not None
        and dependencies.git_port is not None
        and dependencies.process is not None
    ):
        graph = dependencies.graph
        git_port = dependencies.git_port
        process = dependencies.process
        project_root = dependencies.project_root
        handlers = ObserverHandlers(
            session_input=handlers.session_input,
            cancel=lambda run_id: cancel_run(graph, git_port, process, project_root, run_id),
        )
    return WebCommands(
        session_input=handlers.session_input,
        cancel=handlers.cancel,
        graph_edits=_graph_edits(dependencies),
        review_decision=dependencies.review_decision,
        git=dependencies.git,
        owner_capabilities=dependencies.owner_capabilities,
    )
