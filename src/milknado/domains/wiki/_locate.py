"""Filesystem path resolution for wiki roadmap directories.

Separated from _serialize.py because this module has filesystem side effects
(Path.exists() calls), whereas _serialize.py is purely functional.
"""

from __future__ import annotations

from pathlib import Path

from milknado.domains.wiki._serialize import validate_slug


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
