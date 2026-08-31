"""Import a wiki roadmap directory into milknado roadmap + goal nodes.

Seeds the goal layer only: one roadmap node and one goal node per `<goal>.md`,
keyed by a deterministic `wiki_ref` so re-import is idempotent and the same files
yield identical keys across separate dbs. Task decomposition stays a later step
(the existing Planner), so no task nodes are created here.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from milknado.domains.common import NodeKind, NodeSpec, NodeStatus
from milknado.domains.graph import MikadoGraph
from milknado.domains.wiki._locate import (
    read_text,
    resolve_roadmap_dir,
    roadmap_markdown_files,
    write_text_atomic,
)
from milknado.domains.wiki._serialize import (
    Created,
    compute_goal_ref,
    compute_roadmap_ref,
    set_frontmatter_field,
)
from milknado.domains.wiki.model import GoalDocument, Lifecycle, RoadmapModel, load_roadmap


@dataclass(frozen=True)
class ImportResult:
    roadmap_node_id: int
    goal_node_ids: dict[str, int]  # goal slug (= filename stem) -> node id
    created_count: int
    reused_count: int
    stamped: list[str]  # slugs whose missing `created` was stamped on import


@dataclass
class _Counters:
    created: int = 0
    reused: int = 0
    stamped: list[str] = field(default_factory=list)


def _today_iso() -> str:
    return datetime.now(UTC).date().isoformat()


def import_roadmap(
    wiki_root: Path,
    roadmap_slug: str,
    graph: MikadoGraph,
    *,
    today: str | None = None,
) -> ImportResult:
    """Parse a roadmap through the canonical wiki model into graph nodes."""
    roadmap_dir = resolve_roadmap_dir(wiki_root, roadmap_slug)
    stamp = today or _today_iso()
    counters = _Counters()
    _stamp_missing_created(wiki_root, roadmap_dir, stamp, counters)
    model = load_model(roadmap_dir, roadmap_slug)
    roadmap_id, _ = _ingest_file(
        graph,
        kind=NodeKind.ROADMAP,
        parent_id=None,
        created=model.roadmap.created,
        title=roadmap_dir.name,
        ref_for=lambda created: compute_roadmap_ref(roadmap_slug, created),
        counters=counters,
    )
    goal_ids: dict[str, int] = {}
    prereqs: dict[str, tuple[str, ...]] = {}
    documents: list[tuple[int, GoalDocument]] = []
    for goal_slug, document in model.goals.items():
        goal_id, edges = _ingest_file(
            graph,
            kind=NodeKind.GOAL,
            parent_id=roadmap_id,
            created=document.created,
            title=document.title,
            ref_for=lambda created, s=goal_slug: compute_goal_ref(roadmap_slug, s, created),
            counters=counters,
            document=document,
        )
        goal_ids[goal_slug] = goal_id
        prereqs[goal_slug] = edges if document.lifecycle is Lifecycle.ACTIVE else ()
        documents.append((goal_id, document))
    _wire_prereqs(graph, roadmap_slug, goal_ids, prereqs)
    for node_id, document in documents:
        _apply_document_state(graph, node_id, document)
    return ImportResult(
        roadmap_node_id=roadmap_id,
        goal_node_ids=goal_ids,
        created_count=counters.created,
        reused_count=counters.reused,
        stamped=counters.stamped,
    )


def _stamp_missing_created(
    wiki_root: Path, roadmap_dir: Path, stamp: str, counters: _Counters
) -> None:
    for path in roadmap_markdown_files(wiki_root, roadmap_dir):
        text = read_text(wiki_root, path)
        if "created:" in text.split("\n---\n", 1)[0]:
            continue
        updated = set_frontmatter_field(text, "created", stamp)
        write_text_atomic(wiki_root, path, updated)
        counters.stamped.append(path.stem)


def load_model(roadmap_dir: Path, roadmap_slug: str) -> RoadmapModel:
    """Load the canonical model; roadmap_slug remains for signature stability."""
    del roadmap_slug
    return load_roadmap(roadmap_dir)


def _ingest_file(
    graph: MikadoGraph,
    *,
    kind: NodeKind,
    parent_id: int | None,
    created: Created,
    title: str,
    ref_for: Callable[[Created], str],
    counters: _Counters,
    document: GoalDocument | None = None,
) -> tuple[int, tuple[str, ...]]:
    """Find-or-create a graph node from one validated canonical document."""
    ref = ref_for(created)
    existing = graph.find_node_by_wiki_ref(ref)
    if existing is not None:
        counters.reused += 1
        node_id = existing.id
    else:
        node = graph.add_node(title, parent_id=parent_id, spec=NodeSpec(kind=kind, wiki_ref=ref))
        counters.created += 1
        node_id = node.id
    return node_id, document.edges if document is not None else ()


def _apply_document_state(graph: MikadoGraph, node_id: int, document: GoalDocument) -> None:
    """Apply the authored lifecycle transition."""
    if document.lifecycle is not Lifecycle.DEPRECATED:
        return
    current = graph.get_node(node_id)
    if current is None:
        raise ValueError(f"goal {document.slug!r} was not created")
    for child in graph.get_children(node_id, include_archived=True):
        if child.kind is NodeKind.GOAL and child.parent_id != node_id:
            _ = graph.remove_edge(node_id, child.id)
    if current.status is not NodeStatus.DONE:
        _ = graph.set_todo_status(node_id, NodeStatus.DONE)
        current = graph.get_node(node_id)
    if current is None:
        raise ValueError(f"goal {document.slug!r} was not created")
    _ = graph.archive_subtree(node_id)


def _wire_prereqs(
    graph: MikadoGraph,
    roadmap_slug: str,
    goal_ids: dict[str, int],
    prereqs: dict[str, tuple[str, ...]],
) -> None:
    """Add prereq edges (child = prerequisite), skipping any that already exist."""
    for goal_slug, prereq_slugs in prereqs.items():
        goal_id = goal_ids[goal_slug]
        existing = {c.id for c in graph.get_children(goal_id)}
        for prereq_slug in prereq_slugs:
            prereq_id = goal_ids.get(prereq_slug)
            if prereq_id is None:
                raise ValueError(
                    f"goal {goal_slug!r} lists unknown prereq {prereq_slug!r} "
                    + f"(not a goal in roadmap {roadmap_slug!r})"
                )
            if prereq_id not in existing:
                _ = graph.add_edge(goal_id, prereq_id)
