"""CLI hosts for the local web application."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from threading import Thread
from time import sleep
from typing import TYPE_CHECKING, Annotated

from milknado.app.run_source import ExecutionSnapshotSource
from milknado.cli._helpers import DEFAULT_PROJECT_ROOT, ensure_db, load_or_default, typer_option
from milknado.web import (
    HostDependencies,
    LaunchToken,
    create_app,
    observer_commands,
    owner_commands,
)
from milknado.web.polling import PolledSnapshotSource
from milknado.web.server import ServerOptions, run_server

if TYPE_CHECKING:
    from milknado.domains.common import MilknadoConfig, PluginHook
    from milknado.domains.execution import RunLoopResult

PortOption = Annotated[int, typer_option("--port", min=1, max=65535, help="HTTP port")]
NoOpenOption = Annotated[bool, typer_option("--no-open", help="Do not open a browser")]


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
        commands = observer_commands(dependencies=HostDependencies(graph=graph))
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
    from milknado.app.run import build_execution_controller, resolve_feature_branch

    graph = ensure_db(context.config, context.plugins)
    controller = build_execution_controller(graph, context.config, context.project_root)
    login = LaunchToken()
    app = create_app(controller, owner_commands(controller, HostDependencies(graph=graph)), login)
    server_thread = Thread(
        target=services.server,
        args=(app, login),
        kwargs={"options": ServerOptions(port=options.port, no_open=options.no_open)},
        daemon=True,
    )
    server_thread.start()
    result: RunLoopResult | None = None
    interrupts = 0
    scheduling_stopped = False
    try:
        try:
            result = controller.run(
                feature_branch=resolve_feature_branch(context.project_root),
                strict=options.strict,
                allow_protected=options.allow_protected,
            )
        except KeyboardInterrupt:
            interrupts += 1
            controller.stop_scheduling()
            scheduling_stopped = True
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
    finally:
        graph.close()
    return result


__all__ = [
    "OwnerWebContext",
    "OwnerWebOptions",
    "OwnerWebServices",
    "run_owner_web",
    "web",
]
