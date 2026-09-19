"""Run and scheduling command endpoints."""

from __future__ import annotations

from typing import cast

from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route

from milknado.web.app import WebContext
from milknado.web.encoding import json_response


def _unavailable(reason: str) -> Response:
    return json_response({"reason": reason}, status_code=409)


async def cancel_route(request: Request) -> Response:
    context = cast(WebContext, request.app.state.web)  # pyright: ignore[reportAny]
    handler = context.commands.cancel
    if handler is None:
        return _unavailable("Cancel is unavailable.")
    try:
        result = handler(cast(str, request.path_params["run_id"]))
    except ValueError as exc:
        return json_response({"reason": str(exc)}, status_code=404)
    return json_response(result)


async def force_stop_route(request: Request) -> Response:
    context = cast(WebContext, request.app.state.web)  # pyright: ignore[reportAny]
    handler = context.commands.force_stop
    if handler is None:
        return _unavailable("Force stop is unavailable.")
    return json_response(handler(cast(str, request.path_params["run_id"])))


async def stop_scheduling_route(request: Request) -> Response:
    context = cast(WebContext, request.app.state.web)  # pyright: ignore[reportAny]
    handler = context.commands.stop_scheduling
    if handler is None:
        return _unavailable("Stop scheduling is unavailable.")
    handler()
    return json_response({"stopped": True})


ROUTES = (
    Route("/api/runs/{run_id}/cancel", cancel_route, methods=["POST"]),
    Route("/api/runs/{run_id}/force-stop", force_stop_route, methods=["POST"]),
    Route("/api/scheduling/stop", stop_scheduling_route, methods=["POST"]),
)
