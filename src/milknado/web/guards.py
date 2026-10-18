"""Pure ASGI request guards for local web access."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import cast

from starlette.responses import PlainTextResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from milknado.web.login import LaunchToken

LOOPBACK_HOSTS = ("127.0.0.1", "localhost")
_HOSTNAME = re.compile(
    r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?(?:\.[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?)*"
)


def allowed_host(value: str) -> str:
    """Normalize an operator-supplied host name; reject ports, schemes, and paths."""
    host = value.strip().lower().removesuffix(".")
    if _HOSTNAME.fullmatch(host) is None:
        raise ValueError(f"{value!r} is not a host name")
    return host


class RequestGuards:
    def __init__(
        self,
        app: ASGIApp,
        login: LaunchToken,
        allowed_hosts: Iterable[str] = LOOPBACK_HOSTS,
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
        host = authority.split(":", 1)[0].lower()
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
            if origin not in {f"http://{authority}", f"https://{authority}"}:
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
