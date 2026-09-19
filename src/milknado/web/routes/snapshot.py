"""Snapshot endpoint."""

from __future__ import annotations

from typing import cast

import msgspec
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from milknado.web.app import WebContext
from milknado.web.commands import build_capabilities


def snapshot_route(request: Request) -> Response:
    context = cast(WebContext, request.app.state.web)  # pyright: ignore[reportAny]
    payload = cast(dict[str, object], msgspec.to_builtins(context.source.snapshot()))
    capabilities = build_capabilities(context.commands)
    payload["capabilities"] = cast(object, msgspec.to_builtins(capabilities))
    return JSONResponse(payload)


ROUTES = (Route("/api/snapshot", snapshot_route, methods=["GET"]),)
