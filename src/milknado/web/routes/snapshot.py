"""Snapshot endpoint."""

from __future__ import annotations

from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route

from milknado.web.encoding import json_response


def snapshot_route(request: Request) -> Response:
    context = request.app.state.web
    snapshot = context.source.snapshot()
    payload = {
        "goal": snapshot.goal,
        "active_runs": snapshot.active_runs,
        "terminal_runs": snapshot.terminal_runs,
        "completed": snapshot.completed,
        "failed": snapshot.failed,
        "stopped": snapshot.stopped,
        "available": snapshot.available,
        "event_lines": snapshot.event_lines,
        "graph": snapshot.graph,
        "node": snapshot.node,
        "capabilities": context.capabilities,
    }
    return json_response(payload)


ROUTES = (Route("/api/snapshot", snapshot_route, methods=["GET"]),)
