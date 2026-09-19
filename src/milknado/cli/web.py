"""CLI hosts for the local web application."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from threading import Thread
from time import sleep
from typing import Annotated

from milknado.cli._helpers import DEFAULT_PROJECT_ROOT, ensure_db, load_or_default, typer_option
from milknado.web import (
    HostDependencies,
    LaunchToken,
    create_app,
    observer_commands,
    owner_commands,
)
from milknado.web.polling import PolledSnapshotSource
from milknado.web.server import run_server

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
        run_server(app, login, port=port, no_open=no_open)
    finally:
        source.close()
        graph.close()


def _watch_source(project_root: Path, db_path: Path) -> object:
    from milknado.app.watch import WatchSnapshotSource

    return WatchSnapshotSource(project_root, db_path)


def run_owner_web(  # noqa: PLR0913
    project_root: Path,
    config: object,
    plugins: list[object],
    strict: bool,
    allow_protected: bool,
    port: int,
    no_open: bool,
    *,
    server: Callable[..., None] = run_server,
) -> None:
    """Run an execution controller beside its owner web host."""
    from milknado.app.run import build_execution_controller, resolve_feature_branch

    graph = ensure_db(config, plugins)  # type: ignore[arg-type]
    controller = build_execution_controller(graph, config, project_root)  # type: ignore[arg-type]
    login = LaunchToken()
    app = create_app(controller, owner_commands(controller, HostDependencies(graph=graph)), login)
    server_thread = Thread(
        target=server,
        args=(app, login),
        kwargs={"port": port, "no_open": no_open},
        daemon=True,
    )
    server_thread.start()
    try:
        try:
            controller.run(
                feature_branch=resolve_feature_branch(project_root),
                strict=strict,
                allow_protected=allow_protected,
            )
        except KeyboardInterrupt:
            controller.stop_scheduling()
            while server_thread.is_alive():
                try:
                    sleep(0.1)
                except KeyboardInterrupt:
                    break
        while server_thread.is_alive():
            try:
                sleep(0.1)
            except KeyboardInterrupt:
                break
    finally:
        graph.close()


__all__ = ["run_owner_web", "web"]
