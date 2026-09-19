"""Local-only uvicorn hosting for the web adapter."""

from __future__ import annotations

import webbrowser
from collections.abc import Callable
from dataclasses import dataclass
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


@dataclass(frozen=True, slots=True)
class ServerServices:
    opener: Callable[[str], object] = webbrowser.open
    runner: ServerRunner | None = None


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
        services.opener(browser_url)
    (services.runner or UvicornRunner()).run(app, host="127.0.0.1", port=options.port)


class UvicornRunner:
    def run(self, app: Starlette, *, host: str, port: int) -> None:
        uvicorn.run(app, host=host, port=port)


__all__ = [
    "ServerOptions",
    "ServerRunner",
    "ServerServices",
    "UvicornRunner",
    "run_server",
]
