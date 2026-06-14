"""resolve_roadmap_dir: nested vs flat layout resolution.

Pins that nested `roadmaps/<slug>/` wins over flat `<slug>/` when both exist,
flat `<slug>/` is found when no nested dir is present, and FileNotFoundError
names both tried paths when neither has an index.md.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from milknado.domains.wiki._locate import resolve_roadmap_dir


def _make_nested(wiki_root: Path, slug: str) -> Path:
    d = wiki_root / "roadmaps" / slug
    d.mkdir(parents=True)
    (d / "index.md").write_text("---\ncreated: 2026-06-01\n---\n# Nested\n")
    return d


def _make_flat(wiki_root: Path, slug: str) -> Path:
    d = wiki_root / slug
    d.mkdir(parents=True)
    (d / "index.md").write_text("---\ncreated: 2026-06-01\n---\n# Flat\n")
    return d


class TestResolveRoadmapDir:
    def test_nested_found(self, tmp_path: Path) -> None:
        root = tmp_path / "wiki"
        d = _make_nested(root, "my-roadmap")
        assert resolve_roadmap_dir(root, "my-roadmap") == d

    def test_flat_found_when_no_nested(self, tmp_path: Path) -> None:
        root = tmp_path / "wiki"
        d = _make_flat(root, "my-roadmap")
        assert resolve_roadmap_dir(root, "my-roadmap") == d

    def test_nested_wins_over_flat(self, tmp_path: Path) -> None:
        root = tmp_path / "wiki"
        nested = _make_nested(root, "my-roadmap")
        _make_flat(root, "my-roadmap")
        assert resolve_roadmap_dir(root, "my-roadmap") == nested

    def test_not_found_raises_file_not_found(self, tmp_path: Path) -> None:
        root = tmp_path / "wiki"
        root.mkdir()
        with pytest.raises(FileNotFoundError, match="roadmap not found"):
            resolve_roadmap_dir(root, "ghost")

    def test_error_mentions_both_tried_paths(self, tmp_path: Path) -> None:
        root = tmp_path / "wiki"
        root.mkdir()
        with pytest.raises(FileNotFoundError) as exc_info:
            resolve_roadmap_dir(root, "ghost")
        msg = str(exc_info.value)
        assert "roadmaps/ghost" in msg
        assert "ghost/index.md" in msg

    def test_unsafe_slug_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="unsafe roadmap slug"):
            resolve_roadmap_dir(tmp_path, "../escape")

    def test_dir_without_index_not_matched(self, tmp_path: Path) -> None:
        root = tmp_path / "wiki"
        (root / "my-roadmap").mkdir(parents=True)
        # dir exists but has no index.md — should not match
        with pytest.raises(FileNotFoundError):
            resolve_roadmap_dir(root, "my-roadmap")
