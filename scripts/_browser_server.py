"""Minimal background uvicorn server for capture and browser-support scripts.

This mirrors the small server helper in tests/browser/conftest.py, kept separate
so production-adjacent scripts do not import from the test tree.
"""

from __future__ import annotations

import socket
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from threading import Thread
from typing import cast

import uvicorn
from starlette.applications import Starlette

from milknado.web import LaunchToken

BROWSER_TOKEN = "browser-test-token"


def wait_until(predicate: Callable[[], bool], timeout: float = 5.0) -> None:
    """Poll `predicate` until it is true, or raise after `timeout` seconds."""
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise TimeoutError(f"condition was not met within {timeout}s")
        time.sleep(0.02)


def _bound_port(server: uvicorn.Server) -> int:
    address = cast("tuple[str, int]", server.servers[0].sockets[0].getsockname())
    return address[1]


def _port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        return sock.connect_ex(("127.0.0.1", port)) != 0


@dataclass
class BrowserServer:
    """A background uvicorn server that can stop and restart on the same port."""

    app: Starlette
    login: LaunchToken
    port: int = 0
    _server: uvicorn.Server | None = field(default=None, init=False, repr=False)
    _thread: Thread | None = field(default=None, init=False, repr=False)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def login_url(self) -> str:
        return f"{self.base_url}/auth?token={self.login.value}"

    def start(self) -> None:
        config = uvicorn.Config(
            self.app,
            host="127.0.0.1",
            port=self.port,
            log_level="warning",
            timeout_graceful_shutdown=1,
        )
        server = uvicorn.Server(config)
        thread = Thread(target=server.run, daemon=True)
        thread.start()
        deadline = time.monotonic() + 5.0
        while not server.started:
            if time.monotonic() > deadline:
                raise TimeoutError("capture server did not start within 5s")
            time.sleep(0.01)
        self.port = _bound_port(server)
        self._server = server
        self._thread = thread

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._server = None
        self._thread = None
        if self.port:
            wait_until(lambda: _port_is_free(self.port), timeout=5.0)
