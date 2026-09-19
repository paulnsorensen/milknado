"""CLI hosts for the local web application."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Thread
from time import sleep
from typing import TYPE_CHECKING, Annotated, Protocol, cast

from milknado.app.run_source import ExecutionSnapshot, ExecutionSnapshotSource
from milknado.cli._helpers import DEFAULT_PROJECT_ROOT, ensure_db, load_or_default, typer_option
from milknado.web import LaunchToken, create_app, observer_commands, owner_commands
from milknado.web.hosts import HostDependencies
from milknado.web.polling import PolledSnapshotSource
from milknado.web.server import ServerOptions, run_server

if TYPE_CHECKING:
    from milknado.adapters import ChangedFile, GitAdapter
    from milknado.domains.common import GitPort, MilknadoConfig, PluginHook, SessionContext
    from milknado.domains.execution import RunLoopResult
    from milknado.domains.graph import MikadoGraph, OwnerCapabilities

PortOption = Annotated[int, typer_option("--port", min=1, max=65535, help="HTTP port")]
NoOpenOption = Annotated[bool, typer_option("--no-open", help="Do not open a browser")]


class _Controller(Protocol):
    def snapshot(self) -> ExecutionSnapshot: ...

    run: Callable[..., RunLoopResult]

    def stop_scheduling(self) -> None: ...


class _ProjectGitInspection:
    def __init__(self, graph: MikadoGraph, project_root: Path) -> None:
        from milknado.adapters import GitAdapter

        self._graph: MikadoGraph = graph
        self._git: GitAdapter = GitAdapter(project_root)

    @property
    def port(self) -> GitPort:
        return self._git

    def _context(self, run_id: str) -> SessionContext:
        context = self._graph.sessions.view(run_id).context
        if context is None:
            raise ValueError(f"run {run_id!r} has no session context")
        return context

    def changes(self, run_id: str) -> tuple[ChangedFile, ...]:
        return self._git.session_changes(self._context(run_id))

    def diff(self, run_id: str, path: str) -> str:
        return self._git.session_diff(self._context(run_id), path)


def _host_dependencies(
    graph: MikadoGraph,
    config: MilknadoConfig,
    project_root: Path,
    owner: Callable[[], OwnerCapabilities | None] | None = None,
) -> HostDependencies:
    from milknado.adapters import ProcessAdapter

    git = _ProjectGitInspection(graph, project_root)

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


def _owner_capabilities(controller: _Controller, graph: MikadoGraph) -> OwnerCapabilities | None:
    snapshot = controller.snapshot()
    if not snapshot.active_runs:
        return None
    return graph.commands.capabilities(snapshot.active_runs[0].run_id)


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
        commands = observer_commands(dependencies=_host_dependencies(graph, config, project_root))
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


@dataclass(slots=True)
class _ServerTask:
    server: Callable[..., None]
    app: object
    login: LaunchToken
    options: ServerOptions
    errors: list[BaseException]
    ready: Event
    done: Event


@dataclass(slots=True)
class _ControllerTask:
    controller: _Controller
    root: Path
    options: OwnerWebOptions
    results: list[object]


def _serve(task: _ServerTask) -> None:
    task.ready.set()
    try:
        task.server(task.app, task.login, options=task.options)
    except (OSError, RuntimeError, ValueError, TypeError) as exc:
        task.errors.append(exc)
    finally:
        task.done.set()


def _run_controller(task: _ControllerTask) -> None:
    from milknado.app.run import resolve_feature_branch

    try:
        task.results.append(
            task.controller.run(
                feature_branch=resolve_feature_branch(task.root),
                strict=task.options.strict,
                allow_protected=task.options.allow_protected,
            )
        )
    except (OSError, RuntimeError, ValueError, TypeError) as exc:
        task.results.append(exc)


def _wait_for_shutdown(
    controller: _Controller,
    server_thread: Thread,
    controller_thread: Thread,
) -> int:
    interrupts = 0
    scheduling_stopped = False
    while server_thread.is_alive():
        try:
            sleep(0.1)
        except KeyboardInterrupt:
            interrupts += 1
            if interrupts == 1 and not scheduling_stopped:
                controller.stop_scheduling()
                scheduling_stopped = True
            elif interrupts >= 2:
                break
        if not controller_thread.is_alive() and interrupts == 0:
            continue

    return interrupts


@dataclass(slots=True)
class _OwnerLaunch:
    controller: _Controller
    context: OwnerWebContext
    options: OwnerWebOptions
    services: OwnerWebServices
    app: object
    login: LaunchToken


def _start_owner_tasks(
    launch: _OwnerLaunch,
) -> tuple[list[BaseException], list[object], Thread, Thread]:
    errors: list[BaseException] = []
    server_ready = Event()
    server_done = Event()
    server_thread = Thread(
        target=_serve,
        args=(
            _ServerTask(
                launch.services.server,
                launch.app,
                launch.login,
                ServerOptions(port=launch.options.port, no_open=launch.options.no_open),
                errors,
                server_ready,
                server_done,
            ),
        ),
        daemon=True,
    )
    server_thread.start()
    _ = server_ready.wait(timeout=1.0)
    if server_done.is_set() and errors:
        raise errors[0]
    results: list[object] = []
    controller_thread = Thread(
        target=_run_controller,
        args=(
            _ControllerTask(
                launch.controller, launch.context.project_root, launch.options, results
            ),
        ),
        daemon=True,
    )
    controller_thread.start()
    return errors, results, server_thread, controller_thread


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
    controller = build_execution_controller(graph, context.config, context.project_root)
    login = LaunchToken()

    def owner() -> OwnerCapabilities | None:
        return _owner_capabilities(controller, graph)

    dependencies = _host_dependencies(graph, context.config, context.project_root, owner)
    app = create_app(controller, owner_commands(controller, dependencies), login)
    errors, results, server_thread, controller_thread = _start_owner_tasks(
        _OwnerLaunch(controller, context, options, services, app, login)
    )
    try:
        interrupts = _wait_for_shutdown(controller, server_thread, controller_thread)
        if interrupts >= 2:
            if results and not isinstance(results[0], BaseException):
                return cast("RunLoopResult", results[0])
            return None
        if errors:
            raise errors[0]
        if results and isinstance(results[0], BaseException):
            raise results[0]
        return cast("RunLoopResult", results[0]) if results else None
    finally:
        graph.close()


__all__ = [
    "OwnerWebContext",
    "OwnerWebOptions",
    "OwnerWebServices",
    "run_owner_web",
    "web",
]
