# pyright: reportAny=false
"""Graph mutation routes."""

from __future__ import annotations

from json import JSONDecodeError
from typing import cast

import msgspec
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route

from milknado.domains.common import MikadoNode
from milknado.web.app import WebContext
from milknado.web.commands import GraphEditCommands
from milknado.web.encoding import json_response
from milknado.web.node_requests import (
    AddNodeBody,
    EditNodeBody,
    MoveNodeBody,
    NodeRequestContext,
    add_inputs,
    decode_body,
    edit_inputs,
)


def _graph(request: Request) -> GraphEditCommands:
    context = cast(WebContext, request.app.state.web)
    if context.commands.graph_edits is None:
        raise RuntimeError("Graph edits are unavailable.")
    return context.commands.graph_edits


def _node_response(node: MikadoNode) -> Response:
    return json_response(msgspec.to_builtins(node))


def _error(error: Exception) -> Response:
    return json_response({"error": str(error)}, status_code=409)


def _bad_request(error: Exception) -> Response:
    return json_response({"error": str(error)}, status_code=400)


def _add_node(commands: GraphEditCommands, body: AddNodeBody) -> MikadoNode:
    context = NodeRequestContext(commands.project_root, commands.flavor_registry)
    spec, files = add_inputs(body, context)
    return commands.graph.add_node(body.description, body.parent_id, spec, files)


async def add_node(request: Request) -> Response:
    try:
        body = decode_body(await request.json(), AddNodeBody)
        assert isinstance(body, AddNodeBody)
        node = await run_in_threadpool(_add_node, _graph(request), body)
        return _node_response(node)
    except (JSONDecodeError, msgspec.DecodeError) as error:
        return _bad_request(error)
    except (ValueError, AssertionError) as error:
        return _error(error)


def _edit_node(commands: GraphEditCommands, node_id: int, body: EditNodeBody) -> MikadoNode:
    context = NodeRequestContext(commands.project_root, commands.flavor_registry)
    files, artifact = edit_inputs(body, context)
    return commands.graph.edit_node(
        node_id,
        body.description,
        body.kind,
        body.flavor,
        artifact,
        files,
        commands.flavor_registry,
    )


async def edit_node(request: Request) -> Response:
    try:
        body = decode_body(await request.json(), EditNodeBody)
        assert isinstance(body, EditNodeBody)
        node = await run_in_threadpool(
            _edit_node, _graph(request), int(request.path_params["node_id"]), body
        )
        return _node_response(node)
    except (JSONDecodeError, msgspec.DecodeError) as error:
        return _bad_request(error)
    except (ValueError, AssertionError) as error:
        return _error(error)


def _move_node(commands: GraphEditCommands, node_id: int, new_parent_id: int | None) -> MikadoNode:
    if commands.graph.get_node(node_id) is None:
        raise ValueError(f"node {node_id} not found")
    commands.graph.move_node(node_id, new_parent_id)
    node = commands.graph.get_node(node_id)
    assert node is not None
    return node


async def move_node(request: Request) -> Response:
    try:
        body = decode_body(await request.json(), MoveNodeBody)
        assert isinstance(body, MoveNodeBody)
        node = await run_in_threadpool(
            _move_node, _graph(request), int(request.path_params["node_id"]), body.new_parent_id
        )
        return _node_response(node)
    except (JSONDecodeError, msgspec.DecodeError) as error:
        return _bad_request(error)
    except (ValueError, AssertionError) as error:
        return _error(error)


def _archive_node(commands: GraphEditCommands, node_id: int) -> tuple[int, MikadoNode]:
    count = commands.graph.archive_subtree(node_id)
    node = commands.graph.get_node(node_id)
    if node is None:
        raise ValueError(f"node {node_id} not found")
    return count, node


async def archive_node(request: Request) -> Response:
    try:
        count, node = await run_in_threadpool(
            _archive_node, _graph(request), int(request.path_params["node_id"])
        )
        return json_response({"archived": count, "node": msgspec.to_builtins(node)})
    except ValueError as error:
        return _error(error)


ROUTES = (
    Route("/api/nodes", add_node, methods=["POST"]),
    Route("/api/nodes/{node_id:int}", edit_node, methods=["PATCH"]),
    Route("/api/nodes/{node_id:int}/move", move_node, methods=["POST"]),
    Route("/api/nodes/{node_id:int}/archive", archive_node, methods=["POST"]),
)
