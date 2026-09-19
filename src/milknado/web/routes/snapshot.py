"""Snapshot endpoint."""

from __future__ import annotations

from typing import cast

import msgspec
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route

from milknado.web.app import WebContext
from milknado.web.commands import build_capabilities
from milknado.web.encoding import json_response


def snapshot_route(request: Request) -> Response:
    context = cast(WebContext, request.app.state.web)  # pyright: ignore[reportAny]
    snapshot = context.source.snapshot()
    payload = cast(dict[str, object], msgspec.to_builtins(snapshot))
    payload["capabilities"] = build_capabilities(context.commands)
    return json_response(payload)


ROUTES = (Route("/api/snapshot", snapshot_route, methods=["GET"]),)
