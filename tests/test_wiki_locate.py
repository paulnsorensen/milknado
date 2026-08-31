"""resolve_roadmap_dir: nested vs flat layout resolution.

Pins that nested `roadmaps/<slug>/` wins over flat `<slug>/` when both exist,
flat `<slug>/` is found when no nested dir is present, and FileNotFoundError
names both tried paths when neither has an index.md.
"""

from __future__ import annotations

import stat
from pathlib import Path

import pytest

from milknado.domains.wiki._locate import (
    RoadmapPathError,
    locate_roadmap_dir,
    read_text,
    resolve_roadmap_dir,
    write_text_atomic,
)
from milknado.domains.wiki._locate import (
    wiki_root as project_wiki_root,
)
from milknado.domains.wiki._serialize import compute_roadmap_ref


def _make_nested(wiki_root: Path, slug: str) -> Path:
    d = wiki_root / "roadmaps" / slug
    d.mkdir(parents=True)
    _ = (d / "index.md").write_text("---\ncreated: 2026-06-01\n---\n# Nested\n")
    return d


def _make_flat(wiki_root: Path, slug: str) -> Path:
    d = wiki_root / slug
    d.mkdir(parents=True)
    _ = (d / "index.md").write_text("---\ncreated: 2026-06-01\n---\n# Flat\n")
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
        _ = _make_flat(root, "my-roadmap")
        assert resolve_roadmap_dir(root, "my-roadmap") == nested

    def test_not_found_raises_file_not_found(self, tmp_path: Path) -> None:
        root = tmp_path / "wiki"
        root.mkdir()
        with pytest.raises(FileNotFoundError, match="roadmap not found"):
            _ = resolve_roadmap_dir(root, "ghost")

    def test_error_mentions_both_tried_paths(self, tmp_path: Path) -> None:
        root = tmp_path / "wiki"
        root.mkdir()
        with pytest.raises(FileNotFoundError) as exc_info:
            _ = resolve_roadmap_dir(root, "ghost")
        msg = str(exc_info.value)
        assert "roadmaps/ghost" in msg
        assert "ghost/index.md" in msg

    def test_unsafe_slug_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="unsafe roadmap slug"):
            _ = resolve_roadmap_dir(tmp_path, "../escape")

    def test_dir_without_index_not_matched(self, tmp_path: Path) -> None:
        root = tmp_path / "wiki"
        (root / "my-roadmap").mkdir(parents=True)
        # dir exists but has no index.md — should not match
        with pytest.raises(FileNotFoundError):
            _ = resolve_roadmap_dir(root, "my-roadmap")


def test_project_wiki_root_is_canonical(tmp_path: Path) -> None:
    assert project_wiki_root(tmp_path) == tmp_path / ".hallouminate" / "wiki"


class TestRoadmapIdentityLookup:
    def test_ignores_non_roadmaps_and_missing_indexes(self, tmp_path: Path) -> None:
        root = tmp_path / "wiki"
        container = root / "roadmaps"
        container.mkdir(parents=True)
        _ = (container / "notes.txt").write_text("not a roadmap")
        (container / "unfinished").mkdir()
        valid = _make_nested(root, "valid")
        created = "2026-06-01"

        found, slug = locate_roadmap_dir(root, compute_roadmap_ref("valid", created))

        assert (found, slug) == (valid, "valid")


class TestConfinedFilesystem:
    def test_reads_regular_file(self, tmp_path: Path) -> None:
        root = tmp_path / "wiki"
        root.mkdir()
        target = root / "goal.md"
        _ = target.write_text("contract")

        assert read_text(root, target) == "contract"

    def test_read_rejects_directory_and_closes_descriptor(self, tmp_path: Path) -> None:
        root = tmp_path / "wiki"
        root.mkdir()
        target = root / "goal.md"
        target.mkdir()

        with pytest.raises(RoadmapPathError, match="not a regular file"):
            _ = read_text(root, target)

    def test_atomic_replace_preserves_mode_and_leaves_no_temp(self, tmp_path: Path) -> None:
        root = tmp_path / "wiki"
        root.mkdir()
        target = root / "goal.md"
        _ = target.write_text("old")
        target.chmod(0o600)

        _ = write_text_atomic(root, target, "new")

        assert target.read_text() == "new"
        assert stat.S_IMODE(target.stat().st_mode) == 0o600
        assert list(root.iterdir()) == [target]

    def test_atomic_create_writes_new_regular_file(self, tmp_path: Path) -> None:
        root = tmp_path / "wiki"
        root.mkdir()
        target = root / "goal.md"

        _ = write_text_atomic(root, target, "new")

        assert target.read_text() == "new"
        assert target.is_file()
        assert stat.S_IMODE(target.stat().st_mode) == 0o600

    def test_atomic_replace_rejects_directory_destination(self, tmp_path: Path) -> None:
        root = tmp_path / "wiki"
        root.mkdir()
        target = root / "goal.md"
        target.mkdir()

        with pytest.raises(RoadmapPathError, match="not a regular file"):
            _ = write_text_atomic(root, target, "new")

        assert target.is_dir()

    def test_atomic_replace_rejects_symlink_destination(self, tmp_path: Path) -> None:
        root = tmp_path / "wiki"
        root.mkdir()
        outside = tmp_path / "outside.md"
        _ = outside.write_text("outside")
        target = root / "goal.md"
        target.symlink_to(outside)

        with pytest.raises(RoadmapPathError, match="symlinked roadmap path"):
            _ = write_text_atomic(root, target, "new")

        assert outside.read_text() == "outside"

    def test_atomic_write_failure_removes_temporary_file(self, tmp_path: Path) -> None:
        root = tmp_path / "wiki"
        root.mkdir()
        target = root / "goal.md"

        with pytest.raises(UnicodeEncodeError):
            write_text_atomic(root, target, "\ud800")

        assert list(root.iterdir()) == []

    def test_rejects_symlinked_wiki_root(self, tmp_path: Path) -> None:
        actual = tmp_path / "actual"
        actual.mkdir()
        _ = (actual / "goal.md").write_text("outside")
        root = tmp_path / "wiki"
        root.symlink_to(actual, target_is_directory=True)

        with pytest.raises(RoadmapPathError, match="symlinked wiki root"):
            _ = read_text(root, root / "goal.md")

    def test_rejects_non_directory_wiki_root(self, tmp_path: Path) -> None:
        root = tmp_path / "wiki"
        _ = root.write_text("not a directory")

        with pytest.raises(RoadmapPathError, match="symlinked wiki root"):
            _ = read_text(root, root / "goal.md")

    def test_rejects_symlinked_intermediate_directory(self, tmp_path: Path) -> None:
        root = tmp_path / "wiki"
        root.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        _ = (outside / "goal.md").write_text("outside")
        (root / "roadmaps").symlink_to(outside, target_is_directory=True)

        with pytest.raises(RoadmapPathError, match="symlinked roadmap directory"):
            _ = read_text(root, root / "roadmaps" / "goal.md")

    def test_rejects_regular_file_as_intermediate_directory(self, tmp_path: Path) -> None:
        root = tmp_path / "wiki"
        root.mkdir()
        _ = (root / "roadmaps").write_text("not a directory")

        with pytest.raises(RoadmapPathError, match="roadmap directory"):
            _ = read_text(root, root / "roadmaps" / "goal.md")

    def test_rejects_path_outside_root(self, tmp_path: Path) -> None:
        root = tmp_path / "wiki"
        root.mkdir()
        outside = tmp_path / "outside.md"
        _ = outside.write_text("outside")

        with pytest.raises(RoadmapPathError, match="escapes wiki root"):
            _ = read_text(root, outside)

    def test_rejects_root_itself_as_file(self, tmp_path: Path) -> None:
        root = tmp_path / "wiki"
        root.mkdir()

        with pytest.raises(RoadmapPathError, match="must name a file"):
            _ = read_text(root, root)
