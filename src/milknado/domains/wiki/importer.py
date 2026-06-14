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

from milknado.domains.common import NodeKind
from milknado.domains.graph import MikadoGraph
from milknado.domains.wiki._locate import resolve_roadmap_dir
from milknado.domains.wiki._serialize import (
    compute_goal_ref,
    compute_roadmap_ref,
    extract_title,
    load_frontmatter,
    read_prereqs,
    set_frontmatter_field,
)


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
    """Parse `roadmaps/<roadmap_slug>/` or `<roadmap_slug>/` into nodes; fail fast if absent."""
    roadmap_dir = resolve_roadmap_dir(wiki_root, roadmap_slug)
    index_path = roadmap_dir / "index.md"
    stamp = today or _today_iso()
    counters = _Counters()
    roadmap_id, _ = _ingest_file(
        graph,
        index_path,
        kind=NodeKind.ROADMAP,
        parent_id=None,
        ref_for=lambda created: compute_roadmap_ref(roadmap_slug, created),
        stamp=stamp,
        counters=counters,
    )
    goal_ids: dict[str, int] = {}
    prereqs: dict[str, list[str]] = {}
    for path in sorted(roadmap_dir.glob("*.md")):
        if path.name == "index.md":
            continue
        goal_slug = path.stem
        goal_ids[goal_slug], prereqs[goal_slug] = _ingest_file(
            graph,
            path,
            kind=NodeKind.GOAL,
            parent_id=roadmap_id,
            ref_for=lambda created, s=goal_slug: compute_goal_ref(roadmap_slug, s, created),
            stamp=stamp,
            counters=counters,
        )
    _wire_prereqs(graph, roadmap_slug, goal_ids, prereqs)
    return ImportResult(
        roadmap_node_id=roadmap_id,
        goal_node_ids=goal_ids,
        created_count=counters.created,
        reused_count=counters.reused,
        stamped=counters.stamped,
    )


def _ingest_file(
    graph: MikadoGraph,
    path: Path,
    *,
    kind: NodeKind,
    parent_id: int | None,
    ref_for: Callable[[str], str],
    stamp: str,
    counters: _Counters,
) -> tuple[int, list[str]]:
    """Find-or-create the node for one roadmap/goal file; stamp `created` if absent.

    Returns the node id and the file's declared prereq slugs, parsed from the
    single frontmatter read so the caller need not re-read the file.
    """
    text = path.read_text()
    frontmatter = load_frontmatter(text)
    created = frontmatter.get("created")
    try:
        prereqs = read_prereqs(frontmatter)
    except ValueError as exc:
        raise ValueError(f"{path.name}: {exc}") from exc
    if created is None:
        text = set_frontmatter_field(text, "created", stamp)
        path.write_text(text)
        created = stamp
        counters.stamped.append(path.stem)
    ref = ref_for(str(created))
    existing = graph.find_node_by_wiki_ref(ref)
    if existing is not None:
        counters.reused += 1
        return existing.id, prereqs
    description = extract_title(text) or path.stem
    node = graph.add_node(description, parent_id=parent_id, kind=kind, wiki_ref=ref)
    counters.created += 1
    return node.id, prereqs


def _wire_prereqs(
    graph: MikadoGraph,
    roadmap_slug: str,
    goal_ids: dict[str, int],
    prereqs: dict[str, list[str]],
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
                    f"(not a goal in roadmap {roadmap_slug!r})"
                )
            if prereq_id not in existing:
                graph.add_edge(goal_id, prereq_id)
