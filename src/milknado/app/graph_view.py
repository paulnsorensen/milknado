"""Pure graph projection and compact node inspection formatting."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import TypeVar, cast

from msgspec import structs

from milknado.app.graph_display import STATUS_ICONS
from milknado.app.graph_pagination import related_pages
from milknado.domains.common import MikadoNode
from milknado.domains.graph import (
    ArtifactSnapshot,
    CommandReceipt,
    GraphSnapshot,
    NodeDetailSnapshot,
    NodeSessionSnapshot,
    SnapshotPage,
)

_T = TypeVar("_T")


@dataclass(frozen=True, slots=True)
class GraphTreeEntry:
    """Stable tree data for a node or a non-primary DAG reference."""

    node_id: int
    reference_parent_id: int | None = None

    @property
    def is_reference(self) -> bool:
        return self.reference_parent_id is not None

    @property
    def key(self) -> str:
        prefix = "ref" if self.is_reference else "node"
        parent = f":{self.reference_parent_id}" if self.is_reference else ""
        return f"{prefix}:{self.node_id}{parent}"


@dataclass(frozen=True, slots=True)
class GraphTreeProjection:
    roots: tuple[GraphTreeEntry, ...]
    children: Mapping[GraphTreeEntry, tuple[GraphTreeEntry, ...]]
    nodes: Mapping[int, MikadoNode]


def project_graph(snapshot: GraphSnapshot) -> GraphTreeProjection:
    """Build one primary containment tree plus explicit non-primary references."""
    nodes = {node.id: node for node in snapshot.nodes}
    primary: dict[int, list[GraphTreeEntry]] = {}
    references: dict[int, list[GraphTreeEntry]] = {}
    roots = [GraphTreeEntry(node.id) for node in snapshot.nodes if node.parent_id not in nodes]
    root_ids = {entry.node_id for entry in roots}
    for node in snapshot.nodes:
        if node.parent_id in nodes:
            primary.setdefault(node.parent_id, []).append(GraphTreeEntry(node.id))
        elif node.id not in root_ids:
            roots.append(GraphTreeEntry(node.id))
            root_ids.add(node.id)
    for edge in snapshot.edges:
        child = nodes.get(edge.child_id)
        if child is not None and edge.parent_id in nodes and child.parent_id != edge.parent_id:
            references.setdefault(edge.parent_id, []).append(
                GraphTreeEntry(edge.child_id, edge.parent_id)
            )

    children: dict[GraphTreeEntry, tuple[GraphTreeEntry, ...]] = {}
    for node in snapshot.nodes:
        entry = GraphTreeEntry(node.id)
        children[entry] = tuple((*primary.get(node.id, ()), *references.get(node.id, ())))
    return GraphTreeProjection(tuple(roots), children, nodes)


def validate_plain_text(value: object, field: str = "value") -> str:
    """Reject NUL, escape, and other terminal control characters at the UI boundary."""
    if not isinstance(value, str):
        raise ValueError(f"{field} must be text")
    if any(ord(char) < 32 and char not in "\t\n\r" for char in value):
        raise ValueError(f"{field} must contain plain text")
    return value


def validate_local_artifact_path(value: str | None) -> str | None:
    """Accept only a repository-relative artifact path without traversal."""
    if value is None:
        return None
    text = validate_plain_text(value, "artifact_path")
    path = Path(text)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("artifact_path must be repository-relative")
    return text


def tree_label(entry: GraphTreeEntry, node: MikadoNode) -> str:
    """Return a text-only label with a visible status and reference marker."""
    try:
        description = validate_plain_text(node.description, "description")
    except ValueError:
        description = "<invalid description>"
    prefix = "↳ " if entry.is_reference else ""
    icon = STATUS_ICONS.get(node.status, "?")
    return f"{prefix}{icon} {node.id} {node.kind.value} · {description}"


def _field_value(value: object, name: str, *, artifact_path: bool = False) -> str:
    if value is None:
        return "—"
    try:
        if artifact_path:
            if not isinstance(value, str):
                raise ValueError("artifact_path must be text")
            shown = validate_local_artifact_path(value)
        else:
            shown = validate_plain_text(str(value), name)
    except ValueError as error:
        shown = f"<invalid: {error}>"
    return str(shown)


def _field(name: str, value: object, *, artifact_path: bool = False) -> str:
    return f"{name}: {_field_value(value, name, artifact_path=artifact_path)}"


def _page_header(name: str, page: SnapshotPage[_T]) -> str:
    if page.items is None:
        return f"{name}: {page.state}"
    total = "?" if page.total is None else str(page.total)
    end = page.offset + len(page.items)
    more = ", more available" if page.has_more else ""
    return f"{name}: {page.offset + 1}-{end}/{total} loaded ({page.state}{more})"


def _object_lines(index: int, value: object) -> list[str]:
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        return [
            f"  {index}. {_field_value(key, 'field')}: {_field_value(item, str(key))}"
            for key, item in mapping.items()
        ]
    if not is_dataclass(value):
        return [f"  {index}. {_field_value(value, 'value')}"]
    return [
        f"  {index}. {type(value).__name__}",
        *(
            "     "
            + field.name
            + ": "
            + _field_value(cast(object, getattr(value, field.name)), field.name)
            for field in fields(value)
        ),
    ]


def _record_lines(index: int, value: object) -> list[str]:
    if isinstance(value, str | int):
        return [f"  {index}. {_field_value(value, 'value')}"]
    if isinstance(value, MikadoNode):
        return _object_lines(index, value)
    if isinstance(value, NodeSessionSnapshot):
        lines = _object_lines(index, value)
        return [*lines, *_page_lines("     event_history", value.event_history)]
    if isinstance(value, ArtifactSnapshot):
        lines = [f"  {index}. path: {_field_value(value.path, 'path', artifact_path=True)}"]
        lines.append(f"     content: {_field_value(value.content.state, 'state')}")
        if value.content.value is not None:
            lines.append(f"     content_value: {_field_value(value.content.value, 'content')}")
        return lines
    if isinstance(value, CommandReceipt):
        return [
            f"  {index}. CommandReceipt",
            *(
                (
                    f"     {field.name}: "
                    f"{_field_value(cast(object, getattr(value, field.name)), field.name)}"
                )
                for field in structs.fields(CommandReceipt)
            ),
        ]
    return _object_lines(index, value)


def _page_lines(name: str, page: SnapshotPage[_T], indent: str = "") -> list[str]:
    if page.items is None:
        return [f"{indent}{name}: {page.state}"]
    lines = [f"{indent}{_page_header(name, page)}"]
    for index, item in enumerate(page.items, page.offset + 1):
        lines.extend(f"{indent}{line}" for line in _record_lines(index, item))
    return lines


def _detail_lines(detail: NodeDetailSnapshot) -> list[str]:
    parent = SnapshotPage((detail.parent,), 0, 1, 1, False)
    lines = ["Detail data"]
    for name, page in (
        ("parent", parent),
        ("children", detail.children),
        ("ancestors", detail.ancestors),
        ("prerequisite_ids", detail.prerequisite_ids),
        ("dependent_ids", detail.dependent_ids),
        ("reverse_dependents", detail.reverse_dependents),
        ("owned_files", detail.owned_files),
        ("runs", detail.runs),
        ("reviews", detail.reviews),
        ("sessions", detail.sessions),
        ("receipts", detail.receipts),
        ("artifacts", detail.artifacts),
    ):
        lines.extend(_page_lines(name, cast(SnapshotPage[object], page)))
    lines.extend(
        (
            f"goal_claim: {detail.goal_claim.state}",
            f"goal_claim_value: {_field_value(detail.goal_claim.value, 'goal_claim')}",
        )
    )
    return lines


def detail_navigation_text(detail: NodeDetailSnapshot | None) -> str:
    if detail is None:
        return "Related values: loading…"
    pages = related_pages(detail)
    page = next((item for item in pages if item.items is not None), None)
    if page is None:
        return "Related values: not available"
    page_number = page.offset // max(page.limit, 1) + 1
    controls: list[str] = []
    if any(item.offset for item in pages if item.items is not None):
        controls.append("[ previous")
    if any(item.has_more for item in pages if item.items is not None):
        controls.append("] next")
    related = f"Related values page {page_number}"
    if controls:
        related += " · " + " · ".join(controls)
    histories = [
        session.event_history
        for session in detail.sessions.items or ()
        if session.event_history.items is not None
    ]
    if not histories:
        return related + " · History: not available"
    history = histories[0]
    history_number = history.offset // max(history.limit, 1) + 1
    history_text = f"History page {history_number}"
    if any(item.offset for item in histories):
        history_text += " · ( previous"
    if any(item.has_more for item in histories):
        history_text += " · ) next"
    return related + " · " + history_text


def node_inspector_text(node: MikadoNode | None, detail: NodeDetailSnapshot | None = None) -> str:
    """Render every current node field, including nulls, in compact semantic groups."""
    if node is None:
        return "Node inspector\nNo node selected."
    lines = [
        "Node inspector",
        "",
        "Identity",
        _field("id", node.id),
        _field("kind", node.kind.value),
        _field("status", node.status.value),
        "description:",
        _field("  ", node.description).replace("  : ", "  ", 1),
        "",
        "Execution",
        _field("run_id", node.run_id),
        _field("goal_run_id", node.goal_run_id),
        _field("pid", node.pid),
        _field("flavor", node.flavor),
        _field("oversized", node.oversized),
        _field("batch_index", node.batch_index),
        "",
        "Paths",
        _field("parent_id", node.parent_id),
        _field("worktree_path", node.worktree_path),
        _field("branch_name", node.branch_name),
        _field("wiki_ref", node.wiki_ref),
        _field("github_ref", node.github_ref),
        _field("artifact_path", node.artifact_path, artifact_path=True),
        "",
        "Time",
        _field("created_at", node.created_at),
        _field("dispatched_at", node.dispatched_at),
        _field("completed_at", node.completed_at),
        _field("completion_duration_seconds", node.completion_duration_seconds),
        _field("archived_at", node.archived_at),
    ]
    if detail is not None:
        lines.extend(("", *_detail_lines(detail)))
    else:
        lines.extend(("", "Detail data", "loading…"))
    return "\n".join(lines)
