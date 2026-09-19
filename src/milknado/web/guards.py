"""Pure ASGI request guards for local web access."""

from __future__ import annotations

from collections.abc import Iterable
from typing import cast

from starlette.responses import PlainTextResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from milknado.web.login import LaunchToken


class RequestGuards:
    def __init__(
        self,
        app: ASGIApp,
        login: LaunchToken,
        allowed_hosts: Iterable[str] = ("127.0.0.1", "localhost"),
    ) -> None:
        self.app: ASGIApp = app
        self.login: LaunchToken = login
        self.allowed_hosts: frozenset[str] = frozenset(allowed_hosts)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = dict(cast(list[tuple[bytes, bytes]], scope["headers"]))
        authority = headers.get(b"host", b"").decode()
        host = authority.split(":", 1)[0]
        if host not in self.allowed_hosts:
            await PlainTextResponse("Host is not allowed.", status_code=400)(scope, receive, send)
            return
        method = cast(str, scope["method"])
        path = cast(str, scope["path"])
        if path.startswith("/api/") and not _cookie_valid(headers, self.login):
            await PlainTextResponse("Authentication required.", status_code=401)(
                scope, receive, send
            )
            return
        if method not in {"GET", "HEAD", "OPTIONS"}:
            origin = headers.get(b"origin", b"").decode().rstrip("/")
            expected = f"http://{authority}"
            if origin != expected:
                await PlainTextResponse("Origin is not allowed.", status_code=403)(
                    scope, receive, send
                )
                return
        await self.app(scope, receive, send)


def _cookie_valid(headers: dict[bytes, bytes], login: LaunchToken) -> bool:
    raw = headers.get(b"cookie", b"").decode()
    prefix = f"{login.cookie_name}="
    for part in raw.split(";"):
        candidate = part.strip()
        if candidate.startswith(prefix):
            return login.verify(candidate[len(prefix) :])
    return False
