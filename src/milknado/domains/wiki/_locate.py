"""Filesystem path resolution for wiki roadmap directories.

Separated from _serialize.py because this module has filesystem side effects
(Path.exists() calls), whereas _serialize.py is purely functional.
"""

from __future__ import annotations

from pathlib import Path

from milknado.domains.wiki._serialize import (
    compute_goal_ref,
    compute_roadmap_ref,
    load_frontmatter,
    validate_slug,
)


def resolve_roadmap_dir(wiki_root: Path, slug: str) -> Path:
    """Return the roadmap directory for *slug* under *wiki_root*.

    Tries `wiki_root/roadmaps/<slug>/` first (nested layout), then
    `wiki_root/<slug>/` (flat layout).  The first candidate whose `index.md`
    exists is returned; nested wins when both exist.

    Raises FileNotFoundError naming both tried paths when neither has an index.md.
    Raises ValueError for an unsafe slug (same as validate_slug).
    """
    validate_slug(slug)
    nested = wiki_root / "roadmaps" / slug
    flat = wiki_root / slug
    for candidate in (nested, flat):
        if (candidate / "index.md").exists():
            return candidate
    raise FileNotFoundError(
        f"roadmap not found: tried roadmaps/{slug}/index.md and {slug}/index.md under {wiki_root}"
    )


def locate_roadmap_dir(wiki_root: Path, roadmap_ref: str) -> tuple[Path, str]:
    """Recover (dir, slug) by matching each roadmap index.md's computed key.

    Searches nested `roadmaps/*/index.md` first, then flat `wiki_root/*/index.md`
    siblings (for roadmaps imported from a flat layout).
    """
    candidates: list[Path] = []
    roadmaps = wiki_root / "roadmaps"
    if roadmaps.is_dir():
        candidates.extend(sorted(roadmaps.glob("*/index.md")))
    # flat siblings: wiki_root/*/index.md, skipping the roadmaps/ subdir itself
    candidates.extend(
        p for p in sorted(wiki_root.glob("*/index.md")) if p.parent.name != "roadmaps"
    )
    for index_path in candidates:
        slug = index_path.parent.name
        created = load_frontmatter(index_path.read_text()).get("created")
        if created is not None and compute_roadmap_ref(slug, str(created)) == roadmap_ref:
            return index_path.parent, slug
    raise FileNotFoundError(f"no roadmap dir under {wiki_root} matches wiki_ref {roadmap_ref}")


def goal_file_map(roadmap_dir: Path, roadmap_slug: str) -> dict[str, Path]:
    """Map each goal file's computed wiki_ref to its path."""
    mapping: dict[str, Path] = {}
    for path in sorted(roadmap_dir.glob("*.md")):
        if path.name == "index.md":
            continue
        created = load_frontmatter(path.read_text()).get("created")
        if created is not None:
            mapping[compute_goal_ref(roadmap_slug, path.stem, str(created))] = path
    return mapping
