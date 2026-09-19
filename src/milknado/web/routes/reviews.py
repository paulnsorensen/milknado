# pyright: reportAny=false, reportUnnecessaryCast=false
"""Goal review routes."""

from __future__ import annotations

from json import JSONDecodeError
from typing import cast

import msgspec
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route

from milknado.domains.common import NodeKind
from milknado.domains.graph import GoalReviewDecision, GoalReviewDecisionRequest, MikadoGraph
from milknado.web.app import WebContext
from milknado.web.encoding import json_response


class ReviewDecisionBody(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    decision: GoalReviewDecision
    decided_by: str = "web"
    decided_at: str | None = None


def _bad_request(error: Exception) -> Response:
    return json_response({"error": str(error)}, status_code=400)


def _context(request: Request) -> WebContext:
    return cast(WebContext, request.app.state.web)


def _graph(request: Request) -> MikadoGraph:
    commands = _context(request).commands.graph_edits
    if commands is None:
        raise RuntimeError("Graph edits are unavailable.")
    return commands.graph


def _pending_reviews(request: Request) -> list[object]:
    graph = _graph(request)
    records: dict[int, object] = {}
    for node in graph.get_all_nodes():
        if node.kind is not NodeKind.GOAL:
            continue
        admission = graph.goal_admission(node.id)
        if admission.review_id is None or admission.decision is not GoalReviewDecision.PENDING:
            continue
        record = graph.get_goal_review(admission.review_id)
        if record is not None and record.decision is GoalReviewDecision.PENDING:
            records[record.review_id] = record
    return list(records.values())


async def list_reviews(request: Request) -> Response:
    reviews = await run_in_threadpool(_pending_reviews, request)
    return json_response(msgspec.to_builtins(reviews))


async def decide_review(request: Request) -> Response:
    try:
        payload = msgspec.convert(await request.json(), type=ReviewDecisionBody, strict=True)
        commands = _context(request).commands.review_decision
        if commands is None:
            raise PermissionError("Review decisions are unavailable.")
        body = cast(ReviewDecisionBody, payload)
        result = await run_in_threadpool(
            commands,
            GoalReviewDecisionRequest(
                int(request.path_params["review_id"]), body.decision, body.decided_at
            ),
            decided_by=body.decided_by,
        )
        return json_response(msgspec.to_builtins(result))
    except PermissionError as error:
        return json_response({"error": str(error)}, status_code=403)
    except (JSONDecodeError, msgspec.DecodeError) as error:
        return _bad_request(error)
    except ValueError as error:
        return json_response({"error": str(error)}, status_code=409)


ROUTES = (
    Route("/api/reviews", list_reviews, methods=["GET"]),
    Route("/api/reviews/{review_id:int}/decision", decide_review, methods=["POST"]),
)
