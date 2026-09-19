"""Read-only node detail endpoint."""

from __future__ import annotations

from typing import cast

from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route

from milknado.app.run_source import NodeSnapshotRequest
from milknado.web.app import WebContext
from milknado.web.encoding import json_response


def _query_int(request: Request, name: str, default: int) -> int:
    value = request.query_params.get(name)
    if value is None:
        return default
    try:
        result = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def node_detail_route(request: Request) -> Response:
    context = cast(WebContext, request.app.state.web)  # pyright: ignore[reportAny]
    try:
        query = NodeSnapshotRequest(
            node_id=int(request.path_params["node_id"]),  # pyright: ignore[reportAny]
            request_generation=_query_int(request, "request_generation", 0),
            page=_query_int(request, "page", 0),
            limit=_query_int(request, "limit", 50),
            session_event_page=_query_int(request, "session_event_page", 0),
        )
    except ValueError as exc:
        return json_response({"error": str(exc)}, status_code=400)
    result = context.source.node_snapshot(query)
    if result.detail is None:
        return json_response({"error": f"Node {query.node_id} was not found."}, status_code=404)
    return json_response(result)


ROUTES = (Route("/api/nodes/{node_id:int}", node_detail_route, methods=["GET"]),)
