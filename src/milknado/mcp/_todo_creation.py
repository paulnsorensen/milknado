"""Shared validation and creation for MCP todo write tools."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from milknado.domains.common import (
    MilknadoConfig,
    NodeSpec,
    normalize_hint_paths,
    validate_hint_path,
)
from milknado.domains.graph import MikadoGraph
from milknado.mcp._core import Flavor, Kind, NodeSummary, parse_flavor, parse_kind
from milknado.mcp.todo import node_to_summary


@dataclass(frozen=True)
class TodoRequest:
    description: str
    kind: Kind
    files: list[str] | None
    flavor: Flavor | None
    artifact: str | None
    prereqs: tuple[int, ...] | None
    root: Path


def node_inputs(
    cfg: MilknadoConfig, request: TodoRequest
) -> tuple[NodeSpec, tuple[str, ...] | None]:
    if request.artifact is not None:
        validate_hint_path(request.artifact, request.root, label="artifact")
    files = (
        tuple(normalize_hint_paths(request.files, request.root))
        if request.files is not None
        else None
    )
    flavor = (
        parse_flavor(request.flavor, cfg.flavor_registry) if request.flavor is not None else None
    )
    spec = NodeSpec(
        kind=parse_kind(request.kind),
        flavor=flavor,
        artifact_path=request.artifact,
        prereqs=request.prereqs or (),
        flavor_registry=cfg.flavor_registry,
    )
    return spec, files


def create_todo(
    graph: MikadoGraph, cfg: MilknadoConfig, parent_id: int | None, request: TodoRequest
) -> NodeSummary:
    spec, files = node_inputs(cfg, request)
    node = graph.add_node(request.description, parent_id=parent_id, spec=spec, files=files)
    return node_to_summary(node)
