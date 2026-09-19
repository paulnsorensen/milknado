"""Local-only uvicorn hosting for the web adapter."""

from __future__ import annotations

import webbrowser
from collections.abc import Callable
from dataclasses import dataclass
from threading import Thread
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
