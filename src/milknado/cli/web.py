"""CLI hosts for the local web application."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from threading import Thread
from time import sleep
from typing import TYPE_CHECKING, Annotated, Protocol, cast

from milknado.app.run_source import ExecutionSnapshot, ExecutionSnapshotSource
from milknado.cli._helpers import DEFAULT_PROJECT_ROOT, ensure_db, load_or_default, typer_option
from milknado.web import (
    HostDependencies,
    LaunchToken,
    OwnerLaunch,
    PolledSnapshotSource,
    ServerOptions,
    create_app,
    finish_shutdown,
    observer_commands,
    owner_commands,
    run_server,
    start_owner_tasks,
    wait_for_shutdown,
)

if TYPE_CHECKING:
    from milknado.adapters import ChangedFile, GitAdapter
    from milknado.domains.common import GitPort, MilknadoConfig, PluginHook, SessionContext
    from milknado.domains.execution import RunLoopResult
    from milknado.domains.graph import MikadoGraph, OwnerCapabilities

PortOption = Annotated[int, typer_option("--port", min=1, max=65535, help="HTTP port")]
NoOpenOption = Annotated[bool, typer_option("--no-open", help="Do not open a browser")]


class _SnapshotReader(Protocol):
    def snapshot(self) -> ExecutionSnapshot: ...


class _Controller(_SnapshotReader, Protocol):
    run: Callable[..., RunLoopResult]

    def stop_scheduling(self) -> None: ...


class _ProjectGitInspection:
    def __init__(self, project_root: Path) -> None:
        from milknado.adapters import GitAdapter

        self._git: GitAdapter = GitAdapter(project_root)

    @property
    def port(self) -> GitPort:
        return self._git

    def changes(self, context: SessionContext) -> tuple[ChangedFile, ...]:
        return self._git.session_changes(context)

    def diff(self, context: SessionContext, path: str) -> str:
        return self._git.session_diff(context, path)


def _host_dependencies(
    graph: MikadoGraph,
    config: MilknadoConfig,
    project_root: Path,
    owner: Callable[[str | None], OwnerCapabilities | None] | None = None,
) -> HostDependencies:
    from milknado.adapters import ProcessAdapter

    git = _ProjectGitInspection(project_root)

    return HostDependencies(
        graph=graph,
        flavor_registry=getattr(config, "flavor_registry", frozenset()),
        project_root=project_root,
        git_port=git.port,
        process=ProcessAdapter(),
        review_decision=graph.decide_goal_review,
        git=git,
        owner_capabilities=owner,
    )


def _owner_capabilities(
    source: _SnapshotReader, graph: MikadoGraph, run_id: str | None = None
) -> OwnerCapabilities | None:
    if run_id is not None:
        return graph.commands.capabilities(run_id)
    active_runs = source.snapshot().active_runs
    if len(active_runs) != 1:
        return None
    return graph.commands.capabilities(active_runs[0].run_id)


def web(
    project_root: Annotated[
        Path, typer_option("--project-root", help="Project root directory")
    ] = DEFAULT_PROJECT_ROOT,
    port: PortOption = 8000,
    no_open: NoOpenOption = False,
) -> None:
    """Serve the read-only local web view."""
    project_root = project_root.resolve()
    config, plugins = load_or_default(project_root)
    graph = ensure_db(config, plugins)
    source = PolledSnapshotSource(_watch_source(project_root, config.db_path))
    login = LaunchToken()
    try:
        source.start()
        owner = partial(_owner_capabilities, source, graph)
        commands = observer_commands(
            dependencies=_host_dependencies(graph, config, project_root, owner)
        )
        app = create_app(source, commands, login)
        run_server(app, login, ServerOptions(port=port, no_open=no_open))
    finally:
        source.close()
        graph.close()


def _watch_source(project_root: Path, db_path: Path) -> ExecutionSnapshotSource:
    from milknado.app.watch import WatchSnapshotSource

    return WatchSnapshotSource(project_root, db_path)


@dataclass(frozen=True, slots=True)
class OwnerWebContext:
    project_root: Path
    config: MilknadoConfig
    plugins: list[PluginHook]


@dataclass(frozen=True, slots=True)
class OwnerWebOptions:
    strict: bool = False
    allow_protected: bool = False
    port: int = 8000
    no_open: bool = False


@dataclass(frozen=True, slots=True)
class OwnerWebServices:
    server: Callable[..., None] = run_server


def run_owner_web(
    context: OwnerWebContext,
    options: OwnerWebOptions | None = None,
    services: OwnerWebServices | None = None,
) -> RunLoopResult | None:
    """Run an execution controller beside its owner web host."""
    options = options or OwnerWebOptions()
    services = services or OwnerWebServices()
    from milknado.app.run import build_execution_controller

    graph = ensure_db(context.config, context.plugins)
    controller: _Controller | None = None
    controller_thread: Thread | None = None
    interrupts = 0
    errors: list[BaseException] = []
    try:
        controller = build_execution_controller(graph, context.config, context.project_root)
        login = LaunchToken()

        def owner(run_id: str | None = None) -> OwnerCapabilities | None:
            return _owner_capabilities(controller, graph, run_id)

        dependencies = _host_dependencies(graph, context.config, context.project_root, owner)
        app = create_app(controller, owner_commands(controller, dependencies), login)
        errors, results, server_thread, controller_thread = start_owner_tasks(
            OwnerLaunch(controller, context, options, services, app, login)
        )
        interrupts = _wait_for_shutdown(controller, server_thread, controller_thread)
        if interrupts >= 2:
            return cast("RunLoopResult", results[0]) if results else None
        if errors:
            raise errors[0]
        if results and isinstance(results[0], BaseException):
            raise results[0]
        return cast("RunLoopResult", results[0]) if results else None
    finally:
        if controller_thread is not None:
            assert controller is not None
            finish_shutdown(controller, controller_thread, interrupts, errors)
            graph.close()
        else:
            graph.close()


def _wait_for_shutdown(
    controller: _Controller,
    server_thread: Thread,
    controller_thread: Thread,
) -> int:
    return wait_for_shutdown(controller, server_thread, controller_thread, sleep)


__all__ = [
    "OwnerWebContext",
    "OwnerWebOptions",
    "OwnerWebServices",
    "run_owner_web",
    "web",
]
