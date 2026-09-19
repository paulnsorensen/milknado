"""Validated request models for graph mutations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import msgspec

from milknado.domains.common import (
    BUILTIN_FLAVORS,
    NodeKind,
    NodeSpec,
    normalize_hint_paths,
    validate_hint_path,
)


class AddNodeBody(msgspec.Struct, frozen=True):
    description: str
    parent_id: int | None = None
    kind: NodeKind = NodeKind.TASK
    flavor: str | None = None
    artifact: str | None = None
    files: list[str] | None = None
    prereqs: list[int] | None = None


class EditNodeBody(msgspec.Struct, frozen=True, kw_only=True):
    description: str | None = None
    kind: NodeKind | None = None
    flavor: str | None = None
    artifact: str | None = None
    files: list[str] | None = None


class MoveNodeBody(msgspec.Struct, frozen=True):
    new_parent_id: int | None = None


@dataclass(frozen=True, slots=True)
class NodeRequestContext:
    project_root: Path
    flavor_registry: frozenset[str] = BUILTIN_FLAVORS


def decode_body(payload: object, body_type: type[msgspec.Struct]) -> msgspec.Struct:
    return msgspec.convert(payload, type=body_type, strict=True)


def add_inputs(
    body: AddNodeBody, context: NodeRequestContext
) -> tuple[NodeSpec, tuple[str, ...] | None]:
    if body.artifact is not None:
        validate_hint_path(body.artifact, context.project_root, label="artifact")
    files = (
        tuple(normalize_hint_paths(body.files, context.project_root))
        if body.files is not None
        else None
    )
    spec = NodeSpec(
        kind=body.kind,
        flavor=body.flavor,
        artifact_path=body.artifact,
        prereqs=tuple(body.prereqs or ()),
        flavor_registry=context.flavor_registry,
    )
    return spec, files


def edit_inputs(
    body: EditNodeBody, context: NodeRequestContext
) -> tuple[tuple[str, ...] | None, str | None]:
    if body.artifact is not None:
        validate_hint_path(body.artifact, context.project_root, label="artifact")
    files = (
        tuple(normalize_hint_paths(body.files, context.project_root))
        if body.files is not None
        else None
    )
    return files, body.artifact
