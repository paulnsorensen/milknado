"""Read-only metaharness overview endpoint."""

from __future__ import annotations

from typing import cast

from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route

from milknado.domains.wiki import list_knowledge, wiki_root
from milknado.web.app import WebContext
from milknado.web.commands import MetaharnessInfo, SchedulingControl
from milknado.web.encoding import json_response


def _overview(info: MetaharnessInfo, scheduling: SchedulingControl | None) -> dict[str, object]:
    return {
        "project": {"name": info.project_root.name},
        "concurrency": info.concurrency,
        "scheduling": {
            "available": scheduling is not None,
            "paused": scheduling.scheduling_paused() if scheduling is not None else False,
            "reason": None if scheduling is not None else "Pausing the run loop is unavailable.",
        },
        "flavors": {
            name: {"quality_gates": list(gates)} for name, gates in info.flavor_gates.items()
        },
        "knowledge": list_knowledge(wiki_root(info.project_root)),
    }


async def metaharness_route(request: Request) -> Response:
    context = cast(WebContext, request.app.state.web)  # pyright: ignore[reportAny]
    info = context.commands.metaharness
    if info is None:
        return json_response({"reason": "Metaharness info is unavailable."}, status_code=409)
    return json_response(await run_in_threadpool(_overview, info, context.commands.scheduling))


async def pause_route(request: Request) -> Response:
    return await _set_paused(request, paused=True)


async def resume_route(request: Request) -> Response:
    return await _set_paused(request, paused=False)


async def _set_paused(request: Request, *, paused: bool) -> Response:
    context = cast(WebContext, request.app.state.web)  # pyright: ignore[reportAny]
    scheduling = context.commands.scheduling
    if scheduling is None:
        return json_response({"reason": "Pausing the run loop is unavailable."}, status_code=409)
    await run_in_threadpool(scheduling.set_scheduling_paused, paused)
    return json_response({"paused": await run_in_threadpool(scheduling.scheduling_paused)})


ROUTES = (
    Route("/api/metaharness", metaharness_route, methods=["GET"]),
    Route("/api/scheduling/pause", pause_route, methods=["POST"]),
    Route("/api/scheduling/resume", resume_route, methods=["POST"]),
)
