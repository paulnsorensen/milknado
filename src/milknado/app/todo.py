"""Application-layer todo creation and parent-blocking policy."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from milknado.domains.common import MilknadoConfig, NodeKind

from milknado.domains.common import NodeSpec, normalize_hint_paths, validate_hint_path
from milknado.domains.graph import MikadoGraph

_logger = logging.getLogger(__name__)

_console = None


def _get_console():  # noqa: ANN202
    global _console  # noqa: PLW0603
    if _console is None:
        from rich.console import Console

        _console = Console()
    return _console


@dataclass(frozen=True)
class CreateTodoRequest:
    description: str
    kind: NodeKind
    files: list[str] | None
    flavor: str | None
    artifact: str | None
    prereqs: list[int] | None
    root: Path


def create_todo(
    graph,  # noqa: ANN001
    cfg: MilknadoConfig,
    parent_id: int | None,
    request: CreateTodoRequest,
) -> object:
    """Create a todo node; returns the created MikadoNode."""
    if request.artifact is not None:
        validate_hint_path(request.artifact, request.root, label="artifact")
    files = (
        normalize_hint_paths(request.files, request.root) if request.files is not None else None
    )
    node = graph.add_node(
        request.description,
        parent_id=parent_id,
        spec=NodeSpec(
            kind=request.kind,
            flavor=request.flavor,
            artifact_path=request.artifact,
            prereqs=request.prereqs or (),
            flavor_registry=cfg.flavor_registry,
        ),
    )
    if files is not None:
        graph.set_file_ownership(node.id, files)
    return node


def maybe_block_parent(graph: MikadoGraph, parent: int | None) -> None:
    if parent is None:
        return
    parent_node = graph.get_node(parent)
    if parent_node and parent_node.status.value == "running":
        graph.mark_blocked(parent)
        _get_console().print(f"Parent node {parent} marked as blocked.")


def follow_up_parent_id(graph: MikadoGraph) -> int | None:
    """Parent for an auto-parented follow-up: the worker node's own parent."""
    raw = os.environ.get("MILKNADO_NODE_ID", "").strip()
    if not raw:
        return None
    worker_id = int(raw)
    node = graph.get_node(worker_id)
    if node is None:
        raise ValueError(f"MILKNADO_NODE_ID {worker_id} not found")
    return node.parent_id
