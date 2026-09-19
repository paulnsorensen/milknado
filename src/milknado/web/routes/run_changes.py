"""Read-only session Git inspection endpoints."""

from __future__ import annotations

from typing import cast

from starlette.requests import Request
from starlette.responses import PlainTextResponse, Response
from starlette.routing import Route

from milknado.domains.common import GitOperationError, SessionContext
from milknado.web.app import WebContext
from milknado.web.encoding import json_response


def _run_context(request: Request, run_id: str) -> SessionContext | Response:
    context = cast(WebContext, request.app.state.web)  # pyright: ignore[reportAny]
    snapshot = context.source.snapshot()
    runs = (*snapshot.active_runs, *snapshot.terminal_runs)
    run = next((item for item in runs if item.run_id == run_id), None)
    if run is None:
        return json_response({"error": f"Run {run_id} was not found."}, status_code=404)
    if run.session.context is None:
        return json_response({"error": "Run has no session worktree."}, status_code=409)
    if context.commands.git is None:
        return json_response({"error": "Git inspection is unavailable."}, status_code=409)
    return run.session.context


def changes_route(request: Request) -> Response:
    context = cast(WebContext, request.app.state.web)  # pyright: ignore[reportAny]
    run_id = str(request.path_params["run_id"])  # pyright: ignore[reportAny]
    run_context = _run_context(request, run_id)
    if isinstance(run_context, Response):
        return run_context
    git = context.commands.git
    assert git is not None
    try:
        changes = git.changes(run_context)
    except GitOperationError as exc:
        return json_response({"error": str(exc)}, status_code=409)
    return json_response(changes)


def diff_route(request: Request) -> Response:
    context = cast(WebContext, request.app.state.web)  # pyright: ignore[reportAny]
    run_id = str(request.path_params["run_id"])  # pyright: ignore[reportAny]
    run_context = _run_context(request, run_id)
    if isinstance(run_context, Response):
        return run_context
    git = context.commands.git
    assert git is not None
    path = request.query_params.get("path", "")
    try:
        diff = git.diff(run_context, path)
    except GitOperationError as exc:
        return json_response({"error": str(exc)}, status_code=409)
    except ValueError as exc:
        return json_response({"error": str(exc)}, status_code=400)
    return PlainTextResponse(diff)


ROUTES = (
    Route("/api/runs/{run_id}/changes", changes_route, methods=["GET"]),
    Route("/api/runs/{run_id}/diff", diff_route, methods=["GET"]),
)
