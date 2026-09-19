"""Thread helpers for the owner web host."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from threading import Event, Thread
from typing import Protocol

from milknado.web import LaunchToken
from milknado.web.server import ServerOptions


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
        raise RuntimeError("controller did not stop before shutdown deadline")
