"""Local-only uvicorn hosting for the web adapter."""

from __future__ import annotations

import webbrowser
from collections.abc import Callable
from typing import Protocol

import uvicorn
from starlette.applications import Starlette

from milknado.web.login import LaunchToken


class ServerRunner(Protocol):
    def run(self, app: Starlette, *, host: str, port: int) -> None: ...


def run_server(  # noqa: PLR0913
    app: Starlette,
    login: LaunchToken,
    port: int = 8000,
    no_open: bool = False,
    opener: Callable[[str], object] = webbrowser.open,
    runner: ServerRunner | None = None,
) -> None:
    """Serve on loopback, printing a token URL and opening only its safe form."""
    token_url = f"http://127.0.0.1:{port}/auth?token={login.value}"
    browser_url = f"http://127.0.0.1:{port}/"
    print(token_url, flush=True)
    if not no_open:
        opener(browser_url)
    (runner or UvicornRunner()).run(app, host="127.0.0.1", port=port)


class UvicornRunner:
    def run(self, app: Starlette, *, host: str, port: int) -> None:
        uvicorn.run(app, host=host, port=port)


__all__ = ["ServerRunner", "UvicornRunner", "run_server"]
