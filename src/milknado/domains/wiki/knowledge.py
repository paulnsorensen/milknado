"""Newest-first listing of the project wiki's knowledge pages."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

KNOWLEDGE_LIMIT = 30
SUMMARY_LIMIT = 200
_EXCLUDED_NAMES = frozenset({"index.md", "log.md"})


@dataclass(frozen=True, slots=True)
class KnowledgeEntry:
    path: str
    title: str
    summary: str
    updated_at: str  # noqa: V107


def list_knowledge(root: Path) -> tuple[KnowledgeEntry, ...]:
    """Return up to 30 wiki pages under root, newest modification first."""
    if not root.is_dir():
        return ()
    pages = [
        (path, path.stat().st_mtime)
        for path in root.rglob("*.md")
        if path.name not in _EXCLUDED_NAMES and path.is_file() and not path.is_symlink()
    ]
    pages.sort(key=lambda page: (-page[1], page[0]))
    return tuple(_entry(root, path, mtime) for path, mtime in pages[:KNOWLEDGE_LIMIT])


def _entry(root: Path, path: Path, mtime: float) -> KnowledgeEntry:
    title, summary = _title_and_summary(path)
    return KnowledgeEntry(
        path=path.relative_to(root).as_posix(),
        title=title or path.stem,
        summary=summary,
        updated_at=datetime.fromtimestamp(mtime, UTC).isoformat(),
    )


def _title_and_summary(path: Path) -> tuple[str | None, str]:
    title: str | None = None
    paragraph: list[str] = []
    for line in _body_lines(path.read_text(encoding="utf-8", errors="replace")):
        text = line.strip()
        if text.startswith("#"):
            if title is None and text.startswith("# "):
                title = text[2:].strip()
            if paragraph:
                break
        elif text:
            paragraph.append(text)
        elif paragraph:
            break
    return title, " ".join(paragraph)[:SUMMARY_LIMIT]


def _body_lines(text: str) -> list[str]:
    lines = text.splitlines()
    if lines[:1] != ["---"]:
        return lines
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            return lines[index + 1 :]
    return lines
