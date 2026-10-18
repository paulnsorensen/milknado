"""Operator note, retry, and skip routes."""

from __future__ import annotations

from collections.abc import Callable
from json import JSONDecodeError
from threading import Lock
from typing import TypeVar, cast

import msgspec
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route

from milknado.domains.common import MikadoNode, NodeStatus
from milknado.web.app import WebContext
from milknado.web.commands import GraphEditCommands
from milknado.web.encoding import json_response
from milknado.web.node_requests import NoteBody, RetryBody, decode_body

_RETRYABLE = frozenset({NodeStatus.FAILED, NodeStatus.BLOCKED})
_Body = TypeVar("_Body", NoteBody, RetryBody)
# Each action reads a node and then writes it, so concurrent edits must not interleave.
_NODE_EDITS = Lock()


def _commands(request: Request) -> GraphEditCommands | None:
    context = cast(WebContext, request.app.state.web)  # pyright: ignore[reportAny]
    return context.commands.graph_edits


def _existing(commands: GraphEditCommands, node_id: int) -> MikadoNode:
    node = commands.graph.get_node(node_id)
    if node is None:
        raise ValueError(f"node {node_id} not found")
    return node


def _append_note(commands: GraphEditCommands, node: MikadoNode, text: str) -> None:
    _ = commands.graph.edit_node(
        node.id,
        f"{node.description}\n\nOperator note: {text}",
        None,
        None,
        None,
        None,
        commands.flavor_registry,
    )


def _retry(commands: GraphEditCommands, node_id: int, body: RetryBody) -> MikadoNode:
    node = _existing(commands, node_id)
    if node.status not in _RETRYABLE:
        raise ValueError(f"node {node_id} is {node.status.value}; only failed or blocked retry")
    if body.note is not None:
        _append_note(commands, node, body.note)
    _ = commands.graph.set_todo_status(node_id, NodeStatus.PENDING)
    return _existing(commands, node_id)


def _skip(commands: GraphEditCommands, node_id: int, body: RetryBody) -> MikadoNode:
    node = _existing(commands, node_id)
    if node.status is not NodeStatus.FAILED:
        raise ValueError(f"node {node_id} is {node.status.value}; only a failed node skips")
    reason = f" {body.note}" if body.note is not None else ""
    _append_note(commands, node, f"Skipped by the operator.{reason}")
    _ = commands.graph.set_todo_status(node_id, NodeStatus.PENDING)
    _ = commands.graph.set_todo_status(node_id, NodeStatus.DONE)
    return _existing(commands, node_id)


def _note(commands: GraphEditCommands, node_id: int, body: NoteBody) -> MikadoNode:
    _append_note(commands, _existing(commands, node_id), body.text)
    return _existing(commands, node_id)


def _serialized(
    action: Callable[[GraphEditCommands, int, _Body], MikadoNode],
    commands: GraphEditCommands,
    node_id: int,
    body: _Body,
) -> MikadoNode:
    with _NODE_EDITS:
        return action(commands, node_id, body)


async def _respond(
    request: Request,
    body_type: type[_Body],
    action: Callable[[GraphEditCommands, int, _Body], MikadoNode],
) -> Response:
    commands = _commands(request)
    if commands is None:
        return json_response({"error": "Graph edits are unavailable."}, status_code=409)
    node_id = cast(int, request.path_params["node_id"])
    try:
        body = cast(_Body, decode_body(cast(object, await request.json()), body_type))
    except (JSONDecodeError, UnicodeDecodeError, msgspec.DecodeError) as error:
        return json_response({"error": str(error)}, status_code=400)
    try:
        node = await run_in_threadpool(_serialized, action, commands, node_id, body)
    except ValueError as error:
        return json_response({"error": str(error)}, status_code=409)
    return json_response(node)


async def note_route(request: Request) -> Response:
    return await _respond(request, NoteBody, _note)


async def retry_route(request: Request) -> Response:
    return await _respond(request, RetryBody, _retry)


async def skip_route(request: Request) -> Response:
    return await _respond(request, RetryBody, _skip)


ROUTES = (
    Route("/api/nodes/{node_id:int}/note", note_route, methods=["POST"]),
    Route("/api/nodes/{node_id:int}/retry", retry_route, methods=["POST"]),
    Route("/api/nodes/{node_id:int}/skip", skip_route, methods=["POST"]),
)
