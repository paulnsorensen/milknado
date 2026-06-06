"""Milknado MCP stdio server — graph tools for Cursor, Codex, Gemini CLI, etc."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

from milknado._mcp_core import (
    Flavor,
    Kind,
    TodoStatus,
    _parse_flavor,
    _parse_kind,
    _parse_todo_status,
    mcp,
    open_graph,
    resolve_project_root,
)
from milknado.domains.batching import (
    DUMB_ZONE_BUDGET,
    BatchPlan,
    ChangeDependency,
    FileChange,
    HashAnchors,
    NewRelationship,
    SymbolRef,
)
from milknado.domains.batching.change import RelationshipReason

__all__ = ["main", "mcp", "open_graph", "resolve_project_root"]

_logger = logging.getLogger(__name__)


@mcp.tool()
def milknado_graph_summary(
    project_root: str = "",
    status: TodoStatus | None = None,
    kind: Kind | None = None,
    flavor: Flavor | None = None,
) -> dict:
    """Return Mikado nodes (id, status, description), optionally filtered.

    status, kind, and flavor narrow the listing to matching nodes; an empty "nodes"
    list is returned when no node matches (or the graph is empty).
    """
    want_status = _parse_todo_status(status) if status is not None else None
    want_kind = _parse_kind(kind) if kind is not None else None
    want_flavor = _parse_flavor(flavor) if flavor is not None else None
    root = resolve_project_root(project_root or None)
    graph, _cfg = open_graph(root)
    try:
        nodes = [
            n
            for n in graph.get_all_nodes()
            if (want_status is None or n.status == want_status)
            and (want_kind is None or n.kind == want_kind)
            and (want_flavor is None or n.flavor == want_flavor)
        ]
        return {
            "nodes": [
                {"id": n.id, "status": n.status.value, "description": n.description} for n in nodes
            ]
        }
    finally:
        graph.close()


def _parse_mcp_hash_anchors(raw: object) -> HashAnchors | None:
    if not isinstance(raw, dict):
        return None
    d = cast("dict[str, Any]", raw)
    before = d.get("before")
    after = d.get("after")
    if not isinstance(before, str) or not isinstance(after, str):
        return None
    return HashAnchors(before=before, after=after)


def _parse_mcp_symbols(raw: object, label: str) -> list[SymbolRef]:
    if not isinstance(raw, (list, tuple)):
        raise ValueError(f"{label} must be a list of dicts")
    out: list[SymbolRef] = []
    for i, s in enumerate(raw):
        if not isinstance(s, dict):
            raise ValueError(f"{label}[{i}] must be a dict")
        sd = cast("dict[str, Any]", s)
        name = sd.get("name")
        file = sd.get("file")
        if not isinstance(name, str) or not isinstance(file, str):
            raise ValueError(f"{label}[{i}] must have string 'name' and 'file'")
        if Path(file).is_absolute() or ".." in Path(file).parts:
            raise ValueError(
                f"{label}[{i}].file must be repo-relative without traversal, got {file!r}"
            )
        out.append(SymbolRef(name=name, file=file))
    return out


def _dict_to_file_change(d: dict) -> FileChange:
    # #74: validate id/path with clear ValueError instead of opaque KeyError
    _id = d.get("id")
    if not isinstance(_id, str) or not _id:
        raise ValueError(f"change 'id' must be a non-empty string, got {_id!r}")
    path = d.get("path")
    if not isinstance(path, str) or not path:
        raise ValueError(f"change 'path' must be a non-empty string, got {path!r}")
    # #76: traversal guard on path
    if Path(path).is_absolute() or ".." in Path(path).parts:
        raise ValueError(f"path must be repo-relative without traversal, got {path!r}")
    # #76: traversal guard on symbols[].file
    symbols_list = _parse_mcp_symbols(d.get("symbols") or [], "symbols")
    # #66: carry hash_anchors
    hash_anchors = _parse_mcp_hash_anchors(d.get("hash_anchors"))
    # #66: carry dependencies
    raw_deps = d.get("dependencies") or []
    if not isinstance(raw_deps, (list, tuple)):
        raise ValueError("dependencies must be a list of dicts")
    dependencies_list: list[ChangeDependency] = []
    for i, dep in enumerate(raw_deps):
        if not isinstance(dep, dict):
            raise ValueError(f"dependencies[{i}] must be a dict")
        dep_path = dep.get("path")
        if not isinstance(dep_path, str) or not dep_path:
            raise ValueError(f"dependencies[{i}].path must be a non-empty string")
        dep_syms = _parse_mcp_symbols(dep.get("symbols") or [], f"dependencies[{i}].symbols")
        dep_anchors = _parse_mcp_hash_anchors(dep.get("hash_anchors"))
        dep_reason = dep.get("reason", "")
        dependencies_list.append(
            ChangeDependency(
                path=dep_path,
                symbols=tuple(dep_syms),
                hash_anchors=dep_anchors,
                reason=dep_reason if isinstance(dep_reason, str) else "",
            )
        )
    # #66: carry description
    description = d.get("description", "")
    return FileChange(
        id=_id,
        path=path,
        edit_kind=d.get("edit_kind", "modify"),
        symbols=tuple(symbols_list),
        hash_anchors=hash_anchors,
        dependencies=tuple(dependencies_list),
        depends_on=tuple(d.get("depends_on", [])),
        description=description if isinstance(description, str) else "",
    )


_VALID_REASONS = frozenset({"new_file", "new_import", "new_call", "new_type_use"})


def _dict_to_new_relationship(d: dict) -> NewRelationship:
    reason = d["reason"]
    if not isinstance(reason, str):
        raise ValueError(f"reason must be a string, got {type(reason).__name__!r}")
    if reason not in _VALID_REASONS:
        raise ValueError(f"invalid reason: {reason!r}; expected one of {sorted(_VALID_REASONS)}")
    return NewRelationship(
        source_change_id=d["source_change_id"],
        dependant_change_id=d["dependant_change_id"],
        reason=cast(RelationshipReason, reason),
    )


def _plan_to_dict(plan: BatchPlan) -> dict:
    return {
        "batches": [
            {
                "index": b.index,
                "change_ids": list(b.change_ids),
                "depends_on": list(b.depends_on),
                "oversized": b.oversized,
            }
            for b in plan.batches
        ],
        "spread_report": [
            {"symbol": {"name": ss.symbol.name, "file": ss.symbol.file}, "spread": ss.spread}
            for ss in plan.spread_report
        ],
        "solver_status": plan.solver_status,
    }


def _plan_batches_impl(
    changes: list[dict],
    budget: int,
    project_root: Path,
    new_relationships: list[dict] | None = None,
    *,
    force_single_batch: bool = False,
) -> dict:
    from milknado.adapters.crg import CrgAdapter
    from milknado.domains.batching import MEGA_BATCH_THRESHOLD, plan_batches
    from milknado.domains.common.errors import MegaBatchAborted
    from milknado.domains.planning.manifest import MANIFEST_VERSION, PlanChangeManifest
    from milknado.domains.planning.telemetry import record_batch_snapshot

    file_changes = [_dict_to_file_change(c) for c in changes]
    rels = tuple(_dict_to_new_relationship(r) for r in (new_relationships or []))
    # #71: log CRG failures instead of swallowing them silently
    crg = None
    try:
        crg = CrgAdapter(project_root)
        crg.ensure_graph(project_root)
    except Exception as exc:
        _logger.warning("CRG unavailable for MCP planning, proceeding without graph: %s", exc)
        crg = None
    plan = plan_batches(file_changes, budget, crg=crg, new_relationships=rels, root=project_root)
    # #68: abort oversized single batches (MCP-only guard; CLI reports oversized_count instead)
    n = plan.mega_batch_change_count
    if n is not None and not force_single_batch:
        raise MegaBatchAborted(change_count=n, threshold=MEGA_BATCH_THRESHOLD)
    # #68: record telemetry as the CLI path does
    manifest = PlanChangeManifest(
        manifest_version=MANIFEST_VERSION,
        goal="(mcp)",
        goal_summary="",
        spec_path=None,
        changes=tuple(file_changes),
        new_relationships=rels,
    )
    record_batch_snapshot(project_root, manifest, plan)
    return _plan_to_dict(plan)


@mcp.tool()
def milknado_plan_batches(
    changes: list[dict],
    budget: int = DUMB_ZONE_BUDGET,
    project_root: str = "",
    new_relationships: list[dict] | None = None,
    force_single_batch: bool = False,
) -> dict:
    """Compute token-budgeted, precedence-respecting batches for changes."""
    root = resolve_project_root(project_root or None)
    return _plan_batches_impl(
        changes, budget, root, new_relationships, force_single_batch=force_single_batch
    )


def main() -> None:
    # Importing each tool module registers its @mcp.tool()s on the shared instance.
    from milknado import mcp_ralph, mcp_run, mcp_todo, mcp_todo_mutate  # noqa: F401

    mcp.run()
