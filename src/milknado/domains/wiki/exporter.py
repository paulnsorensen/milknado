"""Harvest milknado execution state back into the wiki goal files.

The export is the milknado-owned half of the membrane: it overwrites only the
`<!-- milknado:harvest -->` block and the `status` / `last_synced` frontmatter,
leaving human-authored Intent/Acceptance byte-identical. A goal node milknado
discovered mid-run (no matching file) gets a fresh goal file created for it.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from milknado.domains.common import MikadoNode, NodeKind, NodeStatus
from milknado.domains.graph import MikadoGraph
from milknado.domains.wiki._serialize import (
    HARVEST_END,
    HARVEST_START,
    compute_goal_ref,
    compute_roadmap_ref,
    load_frontmatter,
    replace_harvest_block,
    set_frontmatter_field,
    slugify,
    validate_slug,
)

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExportResult:
    files_written: int  # existing goal files whose harvest block was rewritten
    files_created: int  # orphan goal nodes given a fresh file
    index_refreshed: bool  # whether `hallouminate index` was invoked


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def export_roadmap(
    graph: MikadoGraph,
    roadmap_node_id: int,
    wiki_root: Path,
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
    roadmap_dir, roadmap_slug = _locate_roadmap_dir(wiki_root, roadmap.wiki_ref)
    file_map = _goal_file_map(roadmap_dir, roadmap_slug)
    now_iso = now or _now_iso()
    written = 0
    created = 0
    for goal in graph.get_children(roadmap_node_id):
        if goal.kind != NodeKind.GOAL:
            continue
        inner = _render_harvest_inner(graph, goal)
        path = file_map.get(goal.wiki_ref) if goal.wiki_ref else None
        if path is not None:
            _rewrite_goal_file(path, inner, goal.status.value, now_iso)
            written += 1
        else:
            _create_orphan_goal_file(graph, roadmap_dir, roadmap_slug, goal, inner, now_iso)
            created += 1
    return ExportResult(
        files_written=written,
        files_created=created,
        index_refreshed=_refresh_index(wiki_root),
    )


def resolve_roadmap_node(graph: MikadoGraph, wiki_root: Path, roadmap_slug: str) -> MikadoNode:
    """Find the milknado roadmap node for a wiki slug via its deterministic key.

    Raises FileNotFoundError (no index.md), ValueError (no `created` to key on),
    or LookupError (roadmap not yet imported) — the CLI/MCP veneer maps these to
    clean errors.
    """
    validate_slug(roadmap_slug)
    index_path = wiki_root / "roadmaps" / roadmap_slug / "index.md"
    if not index_path.exists():
        raise FileNotFoundError(f"roadmap not found: {index_path}")
    created = load_frontmatter(index_path.read_text()).get("created")
    if created is None:
        raise ValueError(f"roadmap {roadmap_slug!r} index.md has no `created`; import it first")
    node = graph.find_node_by_wiki_ref(compute_roadmap_ref(roadmap_slug, str(created)))
    if node is None:
        raise LookupError(f"roadmap {roadmap_slug!r} not imported into milknado yet")
    return node


def _locate_roadmap_dir(wiki_root: Path, roadmap_ref: str) -> tuple[Path, str]:
    """Recover (dir, slug) by matching each roadmap index.md's computed key."""
    roadmaps = wiki_root / "roadmaps"
    if roadmaps.is_dir():
        for index_path in sorted(roadmaps.glob("*/index.md")):
            slug = index_path.parent.name
            created = load_frontmatter(index_path.read_text()).get("created")
            if created is not None and compute_roadmap_ref(slug, str(created)) == roadmap_ref:
                return index_path.parent, slug
    raise FileNotFoundError(f"no roadmap dir under {roadmaps} matches wiki_ref {roadmap_ref}")


def _goal_file_map(roadmap_dir: Path, roadmap_slug: str) -> dict[str, Path]:
    """Map each goal file's computed wiki_ref to its path."""
    mapping: dict[str, Path] = {}
    for path in sorted(roadmap_dir.glob("*.md")):
        if path.name == "index.md":
            continue
        created = load_frontmatter(path.read_text()).get("created")
        if created is not None:
            mapping[compute_goal_ref(roadmap_slug, path.stem, str(created))] = path
    return mapping


def _task_descendants(graph: MikadoGraph, goal_id: int) -> list[MikadoNode]:
    """Collect TASK nodes in the goal's subtree, not crossing into sibling goals."""
    tasks: list[MikadoNode] = []
    stack = list(graph.get_children(goal_id))
    while stack:
        node = stack.pop()
        if node.kind == NodeKind.TASK:
            tasks.append(node)
            stack.extend(graph.get_children(node.id))
    return tasks


def _collect_summaries(graph: MikadoGraph, tasks: list[MikadoNode]) -> list[str]:
    summaries: list[str] = []
    for task in tasks:
        for run in graph.runs_for_node(task.id, terminal_only=True):
            msg = graph.latest_run_message(run["run_id"], "result")
            if msg:
                summaries.append(msg.strip())
    return summaries


def _render_harvest_inner(graph: MikadoGraph, goal: MikadoNode) -> str:
    # No separate `follow-ups:` line (despite the spec's illustrated schema):
    # milknado tracks follow-ups as new graph nodes, not a queryable field, so
    # they already fold into the task done/failed rollup below.
    tasks = _task_descendants(graph, goal.id)
    done = sum(1 for t in tasks if t.status == NodeStatus.DONE)
    failed = sum(1 for t in tasks if t.status == NodeStatus.FAILED)
    lines = [f"result: {goal.status.value} · tasks: {done} done / {failed} failed"]
    summaries = _collect_summaries(graph, tasks)
    if summaries:
        lines.append("summary: " + " | ".join(summaries))
    branches = sorted({t.branch_name for t in tasks if t.branch_name})
    if branches:
        lines.append("branches: " + ", ".join(branches))
    return "\n".join(lines)


def _rewrite_goal_file(path: Path, inner: str, status: str, now_iso: str) -> None:
    text = path.read_text()
    text = replace_harvest_block(text, inner)
    text = set_frontmatter_field(text, "status", status)
    text = set_frontmatter_field(text, "last_synced", f'"{now_iso}"')
    path.write_text(text)


def _create_orphan_goal_file(
    graph: MikadoGraph,
    roadmap_dir: Path,
    roadmap_slug: str,
    goal: MikadoNode,
    inner: str,
    now_iso: str,
) -> None:
    base = slugify(goal.description) or "untitled-goal"
    # Two discovered goals with the same description slugify alike; disambiguate
    # with a numeric suffix (never the internal node id, which must not leak into
    # the wiki) so the second neither overwrites the first's file nor collides on
    # the UNIQUE(wiki_ref) index.
    slug = base
    suffix = 2
    while (roadmap_dir / f"{slug}.md").exists():
        slug = f"{base}-{suffix}"
        suffix += 1
    created = now_iso[:10]
    text = (
        f"---\nkind: goal\nslug: {slug}\nroadmap: {roadmap_slug}\n"
        f"created: {created}\nstatus: {goal.status.value}\n"
        f'last_synced: "{now_iso}"\n---\n'
        f"# {goal.description}\n\n"
        "## Intent\n"
        f"{goal.description}\n\n"
        "## Acceptance\n"
        "- (discovered during execution)\n\n"
        "## Outcome\n"
        f"{HARVEST_START}\n{inner}\n{HARVEST_END}\n"
    )
    (roadmap_dir / f"{slug}.md").write_text(text)
    graph.set_wiki_ref(goal.id, compute_goal_ref(roadmap_slug, slug, created))


def _refresh_index(wiki_root: Path) -> bool:
    """Best-effort `hallouminate index`; never fatal if the CLI is absent."""
    if shutil.which("hallouminate") is None:
        return False
    try:
        completed = subprocess.run(
            ["hallouminate", "index"],
            cwd=str(wiki_root),
            capture_output=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        _logger.debug("hallouminate index failed: %s", exc)
        return False
    return completed.returncode == 0
