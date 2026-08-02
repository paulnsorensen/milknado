"""Harvest execution outcomes into confined wiki roadmap files."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from milknado.domains.common import MikadoNode, NodeKind, slugify
from milknado.domains.graph import MikadoGraph
from milknado.domains.reporting import build_harvest_summary, format_harvest_text
from milknado.domains.wiki._locate import (
    locate_roadmap_dir,
    read_text,
    resolve_roadmap_dir,
    write_text_atomic,
)
from milknado.domains.wiki._serialize import (
    HARVEST_END,
    HARVEST_START,
    compute_goal_ref,
    compute_roadmap_ref,
    load_frontmatter,
    replace_harvest_block,
    set_frontmatter_field,
)
from milknado.domains.wiki.importer import _load_model
from milknado.domains.wiki.ports import WikiIndexerPort, WikiIndexResult


@dataclass(frozen=True)
class ExportResult:
    files_written: int
    files_created: int
    index: WikiIndexResult


@dataclass(frozen=True)
class _ExportContext:
    wiki_root: Path
    roadmap_dir: Path
    roadmap_slug: str
    synced_at: str


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def export_roadmap(
    graph: MikadoGraph,
    roadmap_node_id: int,
    wiki_root: Path,
    indexer: WikiIndexerPort,
    *,
    now: str | None = None,
) -> ExportResult:
    """Write execution outcomes for every goal under the roadmap node."""
    roadmap = graph.get_node(roadmap_node_id)
    if roadmap is None or roadmap.kind != NodeKind.ROADMAP:
        raise ValueError(f"node {roadmap_node_id} is not a roadmap node")
    if roadmap.wiki_ref is None:
        raise ValueError(
            f"roadmap node {roadmap_node_id} has no wiki_ref; import the roadmap first"
        )
    roadmap_dir, roadmap_slug = locate_roadmap_dir(wiki_root, roadmap.wiki_ref)
    context = _ExportContext(wiki_root, roadmap_dir, roadmap_slug, now or _now_iso())
    model = _load_model(roadmap_dir, roadmap_slug)
    files = {
        compute_goal_ref(roadmap_slug, slug, goal.created): roadmap_dir / f"{slug}.md"
        for slug, goal in model.goals.items()
    }
    written = 0
    created = 0
    for goal in graph.get_children(roadmap_node_id, include_archived=True):
        if goal.kind != NodeKind.GOAL:
            continue
        harvest = format_harvest_text(build_harvest_summary(graph, goal))
        path = files.get(goal.wiki_ref) if goal.wiki_ref else None
        if path is not None:
            _rewrite_goal_file(context, path, harvest, goal.status.value)
            written += 1
        else:
            _create_orphan_goal_file(graph, context, goal, harvest)
            created += 1
    return ExportResult(written, created, indexer.refresh(wiki_root))


def resolve_roadmap_node(graph: MikadoGraph, wiki_root: Path, roadmap_slug: str) -> MikadoNode:
    """Find the milknado roadmap node for a wiki slug via its deterministic key."""
    roadmap_dir = resolve_roadmap_dir(wiki_root, roadmap_slug)
    created = load_frontmatter(read_text(wiki_root, roadmap_dir / "index.md")).get("created")
    if created is None:
        raise ValueError(f"roadmap {roadmap_slug!r} index.md has no `created`; import it first")
    node = graph.find_node_by_wiki_ref(compute_roadmap_ref(roadmap_slug, str(created)))
    if node is None:
        raise LookupError(f"roadmap {roadmap_slug!r} not imported into milknado yet")
    return node


def _rewrite_goal_file(context: _ExportContext, path: Path, harvest: str, status: str) -> None:
    text = replace_harvest_block(read_text(context.wiki_root, path), harvest)
    text = set_frontmatter_field(text, "status", status)
    text = set_frontmatter_field(text, "last_synced", f'"{context.synced_at}"')
    write_text_atomic(context.wiki_root, path, text)


def _create_orphan_goal_file(
    graph: MikadoGraph,
    context: _ExportContext,
    goal: MikadoNode,
    harvest: str,
) -> None:
    base = slugify(goal.description) or "untitled-goal"
    slug = _available_slug(context.wiki_root, context.roadmap_dir, base)
    created = context.synced_at[:10]
    text = (
        f"---\nkind: goal\nslug: {slug}\nroadmap: {context.roadmap_slug}\n"
        f"created: {created}\nstatus: {goal.status.value}\n"
        f'last_synced: "{context.synced_at}"\n---\n'
        f"# {goal.description}\n\n"
        "## Intent\n"
        f"{goal.description}\n\n"
        "## Acceptance\n"
        "- (discovered during execution)\n\n"
        "## Outcome\n"
        f"{HARVEST_START}\n{harvest}\n{HARVEST_END}\n"
    )
    write_text_atomic(context.wiki_root, context.roadmap_dir / f"{slug}.md", text)
    graph.set_wiki_ref(
        goal.id,
        compute_goal_ref(context.roadmap_slug, slug, created),
    )


def _available_slug(wiki_root: Path, roadmap_dir: Path, base: str) -> str:
    slug = base
    suffix = 2
    while _confined_file_exists(wiki_root, roadmap_dir / f"{slug}.md"):
        slug = f"{base}-{suffix}"
        suffix += 1
    return slug


def _confined_file_exists(wiki_root: Path, path: Path) -> bool:
    try:
        read_text(wiki_root, path)
    except FileNotFoundError:
        return False
    return True
