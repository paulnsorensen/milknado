from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import cast

from milknado.domains.batching import (
    EditKind,
    FileChange,
    NewRelationship,
    RelationshipReason,
    SymbolRef,
)

MANIFEST_VERSION = "milknado.plan.v2"

_VALID_EDIT_KINDS = frozenset({"add", "modify", "delete", "rename"})
_VALID_RELATIONSHIP_REASONS = frozenset(
    {"new_file", "new_import", "new_call", "new_type_use"},
)
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
    if not isinstance(raw, dict):
        _logger.warning("manifest root is not an object")
        return None
    if raw.get("manifest_version") != MANIFEST_VERSION:
        _logger.warning(
            "manifest_version mismatch: expected %s, got %r",
            MANIFEST_VERSION,
            raw.get("manifest_version"),
        )
        return None
    goal = raw.get("goal")
    if not isinstance(goal, str) or not goal.strip():
        _logger.warning("manifest.goal must be a non-empty string")
        return None
    goal = goal.strip()
    goal_summary = raw.get("goal_summary")
    if not isinstance(goal_summary, str) or not goal_summary.strip():
        _logger.warning("manifest.goal_summary must be a non-empty string")
        return None
    goal_summary = goal_summary.strip()
    raw_spec_path = raw.get("spec_path")
    if raw_spec_path is not None and not isinstance(raw_spec_path, str):
        _logger.warning("manifest.spec_path must be a string or null")
        return None
    spec_path: str | None = raw_spec_path if isinstance(raw_spec_path, str) else None
    changes = _parse_changes(raw.get("changes"))
    if changes is None:
        return None
    relationships = _parse_relationships(
        raw.get("new_relationships", []),
        known_ids={c.id for c in changes},
    )
    if relationships is None:
        return None
    return PlanChangeManifest(
        manifest_version=MANIFEST_VERSION,
        goal=goal,
        goal_summary=goal_summary,
        spec_path=spec_path,
        changes=changes,
        new_relationships=relationships,
    )


def _extract_fenced_json(text: str) -> str | None:
    match = _FENCED_JSON_RE.search(text)
    return match.group(1) if match else None


def _parse_changes(raw: object) -> tuple[FileChange, ...] | None:
    if not isinstance(raw, list):
        _logger.warning("manifest.changes must be a list, got %s", type(raw).__name__)
        return None
    parsed: list[FileChange] = []
    seen_ids: set[str] = set()
    for entry in raw:
        change = _parse_single_change(entry)
        if change is None:
            return None
        if change.id in seen_ids:
            _logger.warning("duplicate change id %r", change.id)
            return None
        seen_ids.add(change.id)
        parsed.append(change)
    for change in parsed:
        for dep in change.depends_on:
            if dep not in seen_ids:
                _logger.warning(
                    "change %r depends_on unknown id %r", change.id, dep,
                )
                return None
    return tuple(parsed)


def _parse_single_change(entry: object) -> FileChange | None:
    if not isinstance(entry, dict):
        _logger.warning("change entry must be an object")
        return None
    raw = cast(dict[str, object], entry)
    cid = raw.get("id")
    path = raw.get("path")
    if not isinstance(cid, str) or not cid:
        _logger.warning("change.id must be a non-empty string")
        return None
    if not isinstance(path, str) or not path:
        _logger.warning("change %r: path must be a non-empty string", cid)
        return None
    edit_kind = raw.get("edit_kind", "modify")
    if edit_kind not in _VALID_EDIT_KINDS:
        _logger.warning("change %r: invalid edit_kind %r", cid, edit_kind)
        return None
    symbols = _parse_symbols(raw.get("symbols", []), cid)
    if symbols is None:
        return None
    raw_deps = raw.get("depends_on", [])
    if not isinstance(raw_deps, list) or not all(
        isinstance(d, str) for d in raw_deps
    ):
        _logger.warning("change %r: depends_on must be a list of strings", cid)
        return None
    depends_on: tuple[str, ...] = tuple(cast(list[str], raw_deps))
    description = raw.get("description")
    if not isinstance(description, str) or not description.strip():
        _logger.warning("change %r: description must be a non-empty string", cid)
        return None
    return FileChange(
        id=cid,
        path=path,
        edit_kind=cast(EditKind, edit_kind),
        symbols=symbols,
        depends_on=depends_on,
        description=description.strip(),
    )


def _parse_symbols(raw: object, cid: str) -> tuple[SymbolRef, ...] | None:
    if not isinstance(raw, list):
        _logger.warning("change %r: symbols must be a list", cid)
        return None
    out: list[SymbolRef] = []
    for sym in raw:
        if not isinstance(sym, dict):
            _logger.warning("change %r: symbol entry must be an object", cid)
            return None
        sym_raw = cast(dict[str, object], sym)
        name = sym_raw.get("name")
        file = sym_raw.get("file")
        if not isinstance(name, str) or not isinstance(file, str):
            _logger.warning(
                "change %r: symbol requires string name and file", cid,
            )
            return None
        out.append(SymbolRef(name=name, file=file))
    return tuple(out)


def _parse_relationships(
    raw: object, *, known_ids: set[str],
) -> tuple[NewRelationship, ...] | None:
    if not isinstance(raw, list):
        _logger.warning("new_relationships must be a list")
        return None
    parsed: list[NewRelationship] = []
    for entry in raw:
        if not isinstance(entry, dict):
            _logger.warning("new_relationship entry must be an object")
            return None
        rel = cast(dict[str, object], entry)
        source = rel.get("source_change_id")
        dependant = rel.get("dependant_change_id")
        reason = rel.get("reason")
        if not isinstance(source, str) or source not in known_ids:
            _logger.warning("new_relationship source %r not in changes", source)
            return None
        if not isinstance(dependant, str) or dependant not in known_ids:
            _logger.warning(
                "new_relationship dependant %r not in changes", dependant,
            )
            return None
        if reason not in _VALID_RELATIONSHIP_REASONS:
            _logger.warning("new_relationship reason %r invalid", reason)
            return None
        parsed.append(
            NewRelationship(
                source_change_id=source,
                dependant_change_id=dependant,
                reason=cast(RelationshipReason, reason),
            ),
        )
    return tuple(parsed)
