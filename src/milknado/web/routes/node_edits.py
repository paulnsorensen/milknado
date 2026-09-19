# pyright: reportAny=false
"""Graph mutation routes."""

from __future__ import annotations

from json import JSONDecodeError
from typing import cast

import msgspec
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


async def add_node(request: Request) -> Response:
    try:
        body = decode_body(await request.json(), AddNodeBody)
        assert isinstance(body, AddNodeBody)
        commands = _graph(request)
        context = NodeRequestContext(commands.project_root, commands.flavor_registry)
        spec, files = add_inputs(body, context)
        node = commands.graph.add_node(body.description, body.parent_id, spec, files)
        return _node_response(node)
    except (JSONDecodeError, msgspec.DecodeError) as error:
        return _bad_request(error)
    except (ValueError, AssertionError) as error:
        return _error(error)


async def edit_node(request: Request) -> Response:
    try:
        body = decode_body(await request.json(), EditNodeBody)
        assert isinstance(body, EditNodeBody)
        commands = _graph(request)
        context = NodeRequestContext(commands.project_root, commands.flavor_registry)
        files, artifact = edit_inputs(body, context)
        no_fields = (
            body.description is None
            and body.kind is None
            and body.flavor is None
            and artifact is None
            and files is None
        )
        if no_fields:
            raise ValueError("nothing to edit")
        edited_values = (body.description, body.kind, body.flavor, artifact)
        if any(value is not None for value in edited_values):
            commands.graph.update_node(
                int(request.path_params["node_id"]),
                body.description,
                body.kind,
                body.flavor,
                artifact,
                commands.flavor_registry,
            )
        if files is not None:
            commands.graph.files.claim(int(request.path_params["node_id"]), list(files))
        node = commands.graph.get_node(int(request.path_params["node_id"]))
        if node is None:
            raise ValueError(f"node {request.path_params['node_id']} not found")
        return _node_response(node)
    except (JSONDecodeError, msgspec.DecodeError) as error:
        return _bad_request(error)
    except (ValueError, AssertionError) as error:
        return _error(error)


async def move_node(request: Request) -> Response:
    try:
        body = decode_body(await request.json(), MoveNodeBody)
        assert isinstance(body, MoveNodeBody)
        commands = _graph(request)
        node_id = int(request.path_params["node_id"])
        commands.graph.move_node(node_id, body.new_parent_id)
        node = commands.graph.get_node(node_id)
        if node is None:
            raise ValueError(f"node {node_id} not found")
        return _node_response(node)
    except (JSONDecodeError, msgspec.DecodeError) as error:
        return _bad_request(error)
    except (ValueError, AssertionError) as error:
        return _error(error)


async def archive_node(request: Request) -> Response:
    try:
        commands = _graph(request)
        node_id = int(request.path_params["node_id"])
        count = commands.graph.archive_subtree(node_id)
        node = commands.graph.get_node(node_id)
        if node is None:
            raise ValueError(f"node {node_id} not found")
        return json_response({"archived": count, "node": msgspec.to_builtins(node)})
    except ValueError as error:
        return _error(error)


ROUTES = (
    Route("/api/nodes", add_node, methods=["POST"]),
    Route("/api/nodes/{node_id:int}", edit_node, methods=["PATCH"]),
    Route("/api/nodes/{node_id:int}/move", move_node, methods=["POST"]),
    Route("/api/nodes/{node_id:int}/archive", archive_node, methods=["POST"]),
)
