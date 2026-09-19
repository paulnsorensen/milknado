"""Local-only uvicorn hosting for the web adapter."""

from __future__ import annotations

import webbrowser
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from threading import Event, Thread
from time import sleep
from typing import Protocol

import uvicorn
from starlette.applications import Starlette

from milknado.web.login import LaunchToken


class ServerRunner(Protocol):
    def run(self, app: Starlette, *, host: str, port: int) -> None: ...


@dataclass(frozen=True, slots=True)
class ServerOptions:
    port: int = 8000
    no_open: bool = False
    started: Callable[[], None] | None = None


@dataclass(frozen=True, slots=True)
class ServerServices:
    opener: Callable[[str], object] = webbrowser.open
    runner: ServerRunner | None = None


def _run_uvicorn(app: Starlette, port: int, started: Callable[[], None] | None) -> None:
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port))

    def notify_ready() -> None:
        while not server.started and not server.should_exit:
            sleep(0.01)
        if server.started and started is not None:
            started()

    watcher = Thread(target=notify_ready, daemon=True)
    watcher.start()
    try:
        server.run()
    finally:
        watcher.join(timeout=1.0)


def run_server(
    app: Starlette,
    login: LaunchToken,
    options: ServerOptions | None = None,
    services: ServerServices | None = None,
) -> None:
    """Serve on loopback, printing a token URL and opening only its safe form."""
    options = options or ServerOptions()
    services = services or ServerServices()
    token_url = f"http://127.0.0.1:{options.port}/auth?token={login.value}"
    browser_url = f"http://127.0.0.1:{options.port}/"
    print(token_url, flush=True)
    if not options.no_open:
        _ = services.opener(browser_url)
    if services.runner is None:
        _run_uvicorn(app, options.port, options.started)
    else:
        services.runner.run(app, host="127.0.0.1", port=options.port)
        if options.started is not None:
            options.started()


__all__ = ["ServerOptions", "ServerRunner", "ServerServices", "run_server"]


class Schedulable(Protocol):
    def stop_scheduling(self) -> None: ...


class ControllerRunner(Schedulable, Protocol):
    def run(self, *, feature_branch: str, strict: bool, allow_protected: bool) -> object: ...


class RunOptions(Protocol):
    @property
    def strict(self) -> bool: ...

    @property
    def allow_protected(self) -> bool: ...

    @property
    def port(self) -> int: ...

    @property
    def no_open(self) -> bool: ...


class OwnerContext(Protocol):
    @property
    def project_root(self) -> Path: ...


class OwnerServices(Protocol):
    @property
    def server(self) -> Callable[..., None]: ...


@dataclass(slots=True)
class ServerTask:
    server: Callable[..., None]
    app: object
    login: LaunchToken
    options: ServerOptions
    errors: list[BaseException]
    ready: Event


@dataclass(slots=True)
class ControllerTask:
    controller: ControllerRunner
    root: Path
    options: RunOptions
    results: list[object]


@dataclass(slots=True)
class OwnerLaunch:
    controller: ControllerRunner
    context: OwnerContext
    options: RunOptions
    services: OwnerServices
    app: object
    login: LaunchToken


def serve(task: ServerTask) -> None:
    options = replace(task.options, started=task.ready.set)
    try:
        task.server(task.app, task.login, options=options)
    except BaseException as exc:  # noqa: BLE001 - capture terminal server failures
        task.errors.append(exc)


def run_controller(task: ControllerTask) -> None:
    from milknado.app.run import resolve_feature_branch

    try:
        task.results.append(
            task.controller.run(
                feature_branch=resolve_feature_branch(task.root),
                strict=task.options.strict,
                allow_protected=task.options.allow_protected,
            )
        )
    except BaseException as exc:  # noqa: BLE001 - preserve controller failure
        task.results.append(exc)


def wait_for_shutdown(
    controller: Schedulable,
    server_thread: Thread,
    controller_thread: Thread,
    sleeper: Callable[[float], None],
) -> int:
    interrupts = 0
    scheduling_stopped = False
    while server_thread.is_alive():
        try:
            sleeper(0.1)
        except KeyboardInterrupt:
            if not controller_thread.is_alive():
                if not scheduling_stopped:
                    controller.stop_scheduling()
                return 1
            interrupts += 1
            if interrupts == 1 and not scheduling_stopped:
                controller.stop_scheduling()
                scheduling_stopped = True
            elif interrupts >= 2:
                break
        if not controller_thread.is_alive() and interrupts == 0:
            continue
    return interrupts


def start_owner_tasks(
    launch: OwnerLaunch,
) -> tuple[list[BaseException], list[object], Thread, Thread]:
    errors: list[BaseException] = []
    server_ready = Event()
    server_thread = Thread(
        target=serve,
        args=(
            ServerTask(
                launch.services.server,
                launch.app,
                launch.login,
                ServerOptions(port=launch.options.port, no_open=launch.options.no_open),
                errors,
                server_ready,
            ),
        ),
        daemon=True,
    )
    server_thread.start()
    if not server_ready.wait(timeout=1.0):
        raise errors[0] if errors else TimeoutError("server did not report readiness")
    results: list[object] = []
    controller_thread = Thread(
        target=run_controller,
        args=(
            ControllerTask(
                launch.controller, launch.context.project_root, launch.options, results
            ),
        ),
        daemon=True,
    )
    controller_thread.start()
    return errors, results, server_thread, controller_thread


def finish_shutdown(
    controller: Schedulable,
    controller_thread: Thread,
    interrupts: int,
    errors: list[BaseException],
) -> None:
    if (controller_thread.is_alive() or errors) and interrupts == 0:
        controller.stop_scheduling()
    controller_thread.join(timeout=1.0)
    if controller_thread.is_alive():
        deadline_error = RuntimeError("controller did not stop before shutdown deadline")
        if errors:
            raise errors[0] from deadline_error
        raise deadline_error
