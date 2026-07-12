"""Read a goal's authored Intent out of the wiki for GitHub-side projection.

Reuses the wiki crossover's public roadmap-dir/goal-file resolution so a
wiki-origin goal's Issue body mirrors the same Intent bytes the wiki owns.
"""

from __future__ import annotations

from pathlib import Path

from milknado.domains.common import MikadoNode
from milknado.domains.wiki import extract_section, locate_roadmap_dir
from milknado.domains.wiki import goal_file_map as _wiki_goal_file_map


def goal_file_map(wiki_root: Path, roadmap_ref: str) -> dict[str, Path]:
    """Map each goal's wiki_ref to its markdown file under the roadmap dir."""
    roadmap_dir, roadmap_slug = locate_roadmap_dir(wiki_root, roadmap_ref)
    return _wiki_goal_file_map(roadmap_dir, roadmap_slug)


def goal_intent(goal: MikadoNode, file_map: dict[str, Path]) -> str:
    """Return the goal's authored Intent section, falling back to its description."""
    path = file_map.get(goal.wiki_ref) if goal.wiki_ref else None
    if path is not None:
        section = extract_section(path.read_text(), "Intent")
        if section:
            return section
    return goal.description
