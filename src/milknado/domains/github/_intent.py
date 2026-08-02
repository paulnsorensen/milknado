"""Read a goal's authored Intent out of the wiki for GitHub-side projection.

Reuses the wiki crossover's public roadmap-dir/goal-file resolution so a
wiki-origin goal's Issue body mirrors the same Intent bytes the wiki owns.
"""

from __future__ import annotations

from pathlib import Path

from milknado.domains.common import MikadoNode
from milknado.domains.wiki import goal_file_map as _wiki_goal_file_map
from milknado.domains.wiki import locate_roadmap_dir, read_text
from milknado.domains.wiki.model import parse_goal_document


def goal_file_map(wiki_root: Path, roadmap_ref: str) -> dict[str, Path]:
    """Map each goal's wiki_ref to its markdown file under the roadmap dir."""
    roadmap_dir, roadmap_slug = locate_roadmap_dir(wiki_root, roadmap_ref)
    return _wiki_goal_file_map(wiki_root, roadmap_dir, roadmap_slug)


def goal_intent(goal: MikadoNode, file_map: dict[str, Path], wiki_root: Path) -> str:
    """Return the goal's authored Intent section, falling back to its description."""
    path = file_map.get(goal.wiki_ref) if goal.wiki_ref else None
    if path is not None:
        intent = parse_goal_document(read_text(wiki_root, path), slug=path.stem).intent
        if intent:
            return intent
    return goal.description
