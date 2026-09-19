"""Pure ASGI request guards for local web access."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from starlette.responses import PlainTextResponse


class RequestGuards:
    def __init__(
        self, app: Any, login: Any, allowed_hosts: Iterable[str] = ("127.0.0.1", "localhost")
    ) -> None:
        self.app = app
        self.login = login
        self.allowed_hosts = frozenset(allowed_hosts)

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        host = headers.get(b"host", b"").decode().split(":", 1)[0]
        if host not in self.allowed_hosts:
            await PlainTextResponse("Host is not allowed.", status_code=400)(scope, receive, send)
            return
        method = scope["method"]
        path = scope["path"]
        if path.startswith("/api/") and not _cookie_valid(headers, self.login):
            await PlainTextResponse("Authentication required.", status_code=401)(
                scope, receive, send
            )
            return
        if method not in {"GET", "HEAD", "OPTIONS"}:
            origin = headers.get(b"origin", b"").decode().rstrip("/")
            expected = f"http://{host}"
            if origin != expected:
                await PlainTextResponse("Origin is not allowed.", status_code=403)(
                    scope, receive, send
                )
                return
        await self.app(scope, receive, send)


def _cookie_valid(headers: dict[bytes, bytes], login: Any) -> bool:
    raw = headers.get(b"cookie", b"").decode()
    expected = f"{login.cookie_name}={login.value}"
    return any(part.strip() == expected for part in raw.split(";"))
