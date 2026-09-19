"""Server-sent snapshot stream."""

from __future__ import annotations

from typing import cast

from sse_starlette.sse import EventSourceResponse
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route

from milknado.web.app import WebContext
from milknado.web.fanout import SnapshotFanout


async def stream_route(request: Request) -> Response:
    app = cast(Starlette, request.app)
    context = cast(WebContext, app.state.web)
    fanout = getattr(app.state, "snapshot_fanout", None)
    if fanout is None:
        fanout = SnapshotFanout(context.source)
        app.state.snapshot_fanout = fanout
    return EventSourceResponse(fanout.events())


ROUTES = (Route("/api/stream", stream_route, methods=["GET"]),)
