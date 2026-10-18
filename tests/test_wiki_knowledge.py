import os
from pathlib import Path

from milknado.domains.wiki import list_knowledge


def _page(root: Path, name: str, text: str, mtime: int) -> None:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(text, encoding="utf-8")
    os.utime(path, (mtime, mtime))


def test_missing_wiki_lists_nothing(tmp_path: Path) -> None:
    assert list_knowledge(tmp_path / "absent") == ()


def test_pages_are_newest_first_with_title_and_summary(tmp_path: Path) -> None:
    _page(tmp_path, "index.md", "# Index\n\nskip", 500)
    _page(tmp_path, "log.md", "# Log\n\nskip", 500)
    _page(tmp_path, "old.md", "# Old Page\n\nFirst line\nsecond line.\n\nOther.", 100)
    _page(tmp_path, "deep/new.md", "---\ntitle: x\n---\n# New Page\n## Sub\nBody text.", 200)
    _page(tmp_path, "notes/no-heading.md", "Just prose.", 150)
    _page(tmp_path, "unclosed.md", "---\n# Unclosed\ntext", 50)
    _page(tmp_path, "empty.md", "", 10)
    (tmp_path / "link.md").symlink_to(tmp_path / "old.md")

    entries = list_knowledge(tmp_path)

    assert [entry.path for entry in entries] == [
        "deep/new.md",
        "notes/no-heading.md",
        "old.md",
        "unclosed.md",
        "empty.md",
    ]
    assert [(entry.title, entry.summary) for entry in entries] == [
        ("New Page", "Body text."),
        ("no-heading", "Just prose."),
        ("Old Page", "First line second line."),
        ("Unclosed", "---"),
        ("empty", ""),
    ]
    assert entries[0].updated_at == "1970-01-01T00:03:20+00:00"


def test_listing_caps_at_thirty_and_truncates_summaries(tmp_path: Path) -> None:
    for index in range(35):
        _page(tmp_path, f"p{index:02}.md", "# T\n\n" + "x" * 300, index)

    entries = list_knowledge(tmp_path)

    assert len(entries) == 30
    assert entries[0].path == "p34.md"
    assert len(entries[0].summary) == 200
