from __future__ import annotations

from asyncio import sleep
from collections.abc import AsyncGenerator
from typing import cast

import msgspec
from sse_starlette.sse import EventSourceResponse
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route

from milknado.domains.coordinator import CoordinatorCommand, CoordinatorSnapshot, StartGoal
from milknado.web.app import WebContext
from milknado.web.commands import CoordinatorPort
from milknado.web.encoding import json_response


def _cursor(request: Request) -> int:
    raw = request.query_params.get("cursor", "0")
    if not raw.isdecimal():
        raise ValueError("cursor must be a non-negative integer")
    return int(raw)


async def sessions_route(request: Request) -> Response:
    context = cast(WebContext, request.app.state.web)  # pyright: ignore[reportAny]
    port = context.commands.coordinator
    if port is None:
        return json_response({"error": "Coordinator is unavailable."}, status_code=409)
    return json_response(await run_in_threadpool(port.list_coordinator_sessions))


async def snapshot_route(request: Request) -> Response:
    try:
        context = cast(WebContext, request.app.state.web)  # pyright: ignore[reportAny]
        port = context.commands.coordinator
        if port is None:
            return json_response({"error": "Coordinator is unavailable."}, status_code=409)
        result = await run_in_threadpool(
            port.read_coordinator_snapshot,
            cast(str, request.path_params["session_id"]),
            _cursor(request),
        )
        return json_response(result)
    except ValueError as error:
        return json_response({"error": str(error)}, status_code=400)
    except KeyError:
        return json_response({"error": "Coordinator session does not exist."}, status_code=404)


async def command_route(request: Request) -> Response:
    context = cast(WebContext, request.app.state.web)  # pyright: ignore[reportAny]
    port = context.commands.coordinator
    if port is None:
        return json_response({"error": "Coordinator is unavailable."}, status_code=409)
    try:
        command = cast(
            CoordinatorCommand, msgspec.json.decode(await request.body(), type=CoordinatorCommand)
        )
        session_id = cast(str, request.path_params.get("session_id", ""))
        if isinstance(command, StartGoal) != (not session_id):
            raise msgspec.ValidationError("start_goal uses /api/coordinators/commands")
        result = await run_in_threadpool(port.send_coordinator_command, session_id, command)
        return json_response(result)
    except msgspec.DecodeError as error:
        return json_response({"error": str(error)}, status_code=400)
    except KeyError:
        return json_response({"error": "Coordinator session does not exist."}, status_code=404)
    except ValueError as error:
        return json_response({"error": str(error)}, status_code=409)


async def stream_route(request: Request) -> Response:
    context = cast(WebContext, request.app.state.web)  # pyright: ignore[reportAny]
    port = context.commands.coordinator
    if port is None:
        return json_response({"error": "Coordinator is unavailable."}, status_code=409)
    try:
        cursor = _cursor(request)
        session_id = cast(str, request.path_params["session_id"])
        initial = cast(
            CoordinatorSnapshot,
            await run_in_threadpool(port.read_coordinator_snapshot, session_id, cursor),
        )
    except ValueError as error:
        return json_response({"error": str(error)}, status_code=400)
    except KeyError:
        return json_response({"error": "Coordinator session does not exist."}, status_code=404)

    return EventSourceResponse(_events(request, port, initial, (session_id, cursor)))


async def _events(
    request: Request,
    port: CoordinatorPort,
    initial: CoordinatorSnapshot,
    position: tuple[str, int],
) -> AsyncGenerator[dict[str, str]]:
    session_id, cursor = position
    snapshot = initial
    while not await request.is_disconnected():
        if snapshot.events:
            cursor = snapshot.cursor
            yield {"event": "coordinator", "data": msgspec.json.encode(snapshot).decode()}
        await sleep(0.5)
        snapshot = cast(
            CoordinatorSnapshot,
            await run_in_threadpool(port.read_coordinator_snapshot, session_id, cursor),
        )


ROUTES = (
    Route("/api/coordinators", sessions_route, methods=["GET"]),
    Route("/api/coordinators/commands", command_route, methods=["POST"]),
    Route("/api/coordinators/{session_id}/commands", command_route, methods=["POST"]),
    Route("/api/coordinators/{session_id}/snapshot", snapshot_route, methods=["GET"]),
    Route("/api/coordinators/{session_id}/stream", stream_route, methods=["GET"]),
)
