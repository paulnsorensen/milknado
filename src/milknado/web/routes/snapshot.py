# pyright: reportAny=false, reportExplicitAny=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnannotatedClassAttribute=false, reportUnnecessaryCast=false, reportUnnecessaryIsInstance=false
"""Snapshot endpoint."""

from __future__ import annotations

from typing import cast

from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route

from milknado.app.run_source import ExecutionSnapshot
from milknado.web.app import WebContext
from milknado.web.encoding import json_response


def snapshot_route(request: Request) -> Response:
    context = cast(WebContext, request.app.state.web)
    snapshot = cast(ExecutionSnapshot, context.source.snapshot())
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
