"""Session input command endpoint."""

from __future__ import annotations

from typing import cast

import msgspec
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route

from milknado.domains.common import SessionInput
from milknado.web.app import WebContext
from milknado.web.encoding import json_response


def _command(body: object) -> SessionInput:
    command = msgspec.convert(body, type=SessionInput, strict=True)
    if not command.command_id.strip():
        raise ValueError("command_id is required")
    if command.action in {"steer", "follow_up"} and not command.text.strip():
        raise ValueError("text is required for message actions")
    if command.action in {"approve", "deny"} and not command.request_id.strip():
        raise ValueError("request_id is required for permission decisions")
    return command


async def session_input_route(request: Request) -> Response:
    context = cast(WebContext, request.app.state.web)  # pyright: ignore[reportAny]
    handler = context.commands.session_input
    if handler is None:
        return json_response({"reason": "Session input is unavailable."}, status_code=409)
    try:
        command = _command(cast(object, await request.json()))
    except (ValueError, TypeError, msgspec.DecodeError) as exc:
        return json_response({"reason": str(exc)}, status_code=400)
    try:
        admitted = await run_in_threadpool(
            handler, cast(str, request.path_params["run_id"]), command
        )
    except ValueError as exc:
        return json_response({"reason": str(exc)}, status_code=409)
    if admitted is None:
        return json_response({"reason": "Session input was rejected."}, status_code=409)
    return json_response(admitted)


ROUTES = (Route("/api/runs/{run_id}/session-input", session_input_route, methods=["POST"]),)
