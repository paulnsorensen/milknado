from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from pydantic import BaseModel, ConfigDict, Field, StrictStr, ValidationError, field_validator

from milknado.domains.batching import (
    ChangeDependency,
    EditKind,
    FileChange,
    HashAnchors,
    NewRelationship,
    RelationshipReason,
    SymbolRef,
)

MANIFEST_VERSION = "milknado.plan.v2"

_FENCED_JSON_RE = re.compile(r"```json\s*\n(.*?)\n```", re.DOTALL)

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PlanChangeManifest:
    manifest_version: str
    goal: str
    goal_summary: str
    spec_path: str | None
    changes: tuple[FileChange, ...]
    new_relationships: tuple[NewRelationship, ...]


class _PlanningModel(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)


def _require_list(value: object) -> object:
    if not isinstance(value, list):
        raise ValueError("must be a list")
    return value


def _validate_relative_path(value: str, *, field_name: str) -> str:
    path = Path(value)
    if not value or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{field_name} must be repo-relative without traversal")
    return value


class _SymbolModel(_PlanningModel):
    name: StrictStr
    file: StrictStr

    @field_validator("file")
    @classmethod
    def validate_file(cls, value: str) -> str:
        return _validate_relative_path(value, field_name="symbol file")


class _HashAnchorsModel(_PlanningModel):
    before: StrictStr
    after: StrictStr

    @field_validator("before", "after")
    @classmethod
    def strip_non_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("hash anchor must be a non-empty string")
        return value


class _DependencyModel(_PlanningModel):
    path: StrictStr
    symbols: list[_SymbolModel] = Field(default_factory=list)
    hash_anchors: _HashAnchorsModel | None = None
    reason: StrictStr = ""

    @field_validator("symbols", mode="before")
    @classmethod
    def validate_symbols(cls, value: object) -> object:
        return _require_list(value)

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("dependency path must be a non-empty string")
        return value

    @field_validator("reason")
    @classmethod
    def strip_reason(cls, value: str) -> str:
        return value.strip()


class _ChangeModel(_PlanningModel):
    id: StrictStr
    path: StrictStr
    edit_kind: EditKind = "modify"
    symbols: list[_SymbolModel] = Field(default_factory=list)
    hash_anchors: _HashAnchorsModel | None = None
    dependencies: list[_DependencyModel] = Field(default_factory=list)
    depends_on: list[StrictStr] = Field(default_factory=list)
    description: StrictStr

    @field_validator("symbols", "dependencies", "depends_on", mode="before")
    @classmethod
    def validate_lists(cls, value: object) -> object:
        return _require_list(value)

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        if not value:
            raise ValueError("change id must be a non-empty string")
        return value

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        return _validate_relative_path(value, field_name="change path")

    @field_validator("description")
    @classmethod
    def strip_description(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("description must be a non-empty string")
        return value


class _RelationshipModel(_PlanningModel):
    source_change_id: StrictStr
    dependant_change_id: StrictStr
    reason: RelationshipReason


class _ManifestModel(_PlanningModel):
    manifest_version: StrictStr
    goal: StrictStr
    goal_summary: StrictStr
    spec_path: StrictStr | None = None
    changes: list[_ChangeModel]
    new_relationships: list[_RelationshipModel] = Field(default_factory=list)

    @field_validator("changes", "new_relationships", mode="before")
    @classmethod
    def validate_lists(cls, value: object) -> object:
        return _require_list(value)

    @field_validator("goal", "goal_summary")
    @classmethod
    def strip_non_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("manifest text must be a non-empty string")
        return value


def _hash_anchors_to_dict(anchors: HashAnchors) -> dict[str, object]:
    return {"before": anchors.before, "after": anchors.after}


def _dependency_to_dict(dependency: ChangeDependency) -> dict[str, object]:
    result: dict[str, object] = {
        "path": dependency.path,
        "symbols": [{"name": s.name, "file": s.file} for s in dependency.symbols],
        "reason": dependency.reason,
    }
    if dependency.hash_anchors is not None:
        result["hash_anchors"] = _hash_anchors_to_dict(dependency.hash_anchors)
    return result


def _change_to_dict(change: FileChange) -> dict[str, object]:
    result: dict[str, object] = {
        "id": change.id,
        "path": change.path,
        "edit_kind": change.edit_kind,
        "description": change.description,
        "symbols": [{"name": s.name, "file": s.file} for s in change.symbols],
        "depends_on": list(change.depends_on),
    }
    if change.hash_anchors is not None:
        result["hash_anchors"] = _hash_anchors_to_dict(change.hash_anchors)
    if change.dependencies:
        result["dependencies"] = [_dependency_to_dict(d) for d in change.dependencies]
    return result


def manifest_to_dict(manifest: PlanChangeManifest) -> dict[str, object]:
    return {
        "manifest_version": manifest.manifest_version,
        "goal": manifest.goal,
        "goal_summary": manifest.goal_summary,
        "spec_path": manifest.spec_path,
        "changes": [_change_to_dict(change) for change in manifest.changes],
        "new_relationships": [
            {
                "source_change_id": relationship.source_change_id,
                "dependant_change_id": relationship.dependant_change_id,
                "reason": relationship.reason,
            }
            for relationship in manifest.new_relationships
        ],
    }


def parse_manifest_from_output(text: str) -> PlanChangeManifest | None:
    block = _extract_fenced_json(text)
    if block is None:
        _logger.warning("manifest missing fenced json block")
        return None
    try:
        raw = json.loads(block)
    except json.JSONDecodeError as exc:
        _logger.warning("manifest json decode failed: %s", exc)
        return None
    return parse_manifest_from_dict(raw)


def decode_manifest(raw: object) -> PlanChangeManifest:
    """Decode a manifest or raise the public boundary error."""
    manifest = _decode_manifest_or_none(raw)
    if manifest is None:
        raise ValueError("manifest is not a valid milknado.plan.v2 object")
    return manifest


def parse_manifest_from_dict(raw: object) -> PlanChangeManifest | None:
    """Decode untrusted planner output without raising."""
    try:
        return decode_manifest(raw)
    except ValueError:
        return None


def _decode_manifest_or_none(raw: object) -> PlanChangeManifest | None:
    if not isinstance(raw, dict):
        _logger.warning("manifest root is not an object")
        return None
    raw_dict = cast(dict[str, object], raw)
    if raw_dict.get("manifest_version") != MANIFEST_VERSION:
        _logger.warning(
            "manifest_version mismatch: expected %s, got %r",
            MANIFEST_VERSION,
            raw_dict.get("manifest_version"),
        )
        return None
    try:
        model = _ManifestModel.model_validate(raw_dict)
    except ValidationError as exc:
        _logger.warning("manifest validation failed: %s", exc)
        return None

    known_ids = {change.id for change in model.changes}
    if len(known_ids) != len(model.changes):
        _logger.warning("duplicate change id in manifest")
        return None
    for change in model.changes:
        for dependency_id in change.depends_on:
            if dependency_id not in known_ids:
                _logger.warning(
                    "change %r depends_on unknown id %r",
                    change.id,
                    dependency_id,
                )
                return None
    for relationship in model.new_relationships:
        if relationship.source_change_id not in known_ids:
            _logger.warning(
                "new_relationship source %r not in changes",
                relationship.source_change_id,
            )
            return None
        if relationship.dependant_change_id not in known_ids:
            _logger.warning(
                "new_relationship dependant %r not in changes",
                relationship.dependant_change_id,
            )
            return None

    return PlanChangeManifest(
        manifest_version=model.manifest_version,
        goal=model.goal,
        goal_summary=model.goal_summary,
        spec_path=model.spec_path,
        changes=tuple(
            FileChange(
                id=change.id,
                path=change.path,
                edit_kind=change.edit_kind,
                symbols=tuple(
                    SymbolRef(name=symbol.name, file=symbol.file) for symbol in change.symbols
                ),
                hash_anchors=(
                    None
                    if change.hash_anchors is None
                    else HashAnchors(
                        before=change.hash_anchors.before,
                        after=change.hash_anchors.after,
                    )
                ),
                dependencies=tuple(
                    ChangeDependency(
                        path=dependency.path,
                        symbols=tuple(
                            SymbolRef(name=symbol.name, file=symbol.file)
                            for symbol in dependency.symbols
                        ),
                        hash_anchors=(
                            None
                            if dependency.hash_anchors is None
                            else HashAnchors(
                                before=dependency.hash_anchors.before,
                                after=dependency.hash_anchors.after,
                            )
                        ),
                        reason=dependency.reason,
                    )
                    for dependency in change.dependencies
                ),
                depends_on=tuple(change.depends_on),
                description=change.description,
            )
            for change in model.changes
        ),
        new_relationships=tuple(
            NewRelationship(
                source_change_id=relationship.source_change_id,
                dependant_change_id=relationship.dependant_change_id,
                reason=relationship.reason,
            )
            for relationship in model.new_relationships
        ),
    )


def _extract_fenced_json(text: str) -> str | None:
    match = _FENCED_JSON_RE.search(text)
    return match.group(1) if match else None
