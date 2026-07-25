from __future__ import annotations

from pathlib import Path

import pytest

from milknado.domains.common.paths import (
    TrustedGlobalPath,
    normalize_hint_paths,
    resolve_project_path,
    slugify,
    trust_global_path,
)


class TestNormalizeHintPaths:
    def test_absolute_under_root_becomes_relative(self, tmp_path: Path) -> None:
        result = normalize_hint_paths([str(tmp_path / "src" / "foo.py")], tmp_path)
        assert result == ["src/foo.py"]

    def test_relative_path_passes_through(self, tmp_path: Path) -> None:
        result = normalize_hint_paths(["src/bar.py"], tmp_path)
        assert result == ["src/bar.py"]

    def test_dotdot_escape_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="escapes"):
            normalize_hint_paths(["../outside.py"], tmp_path)

    def test_absolute_outside_root_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="escapes"):
            normalize_hint_paths(["/etc/passwd"], tmp_path)

    def test_nonexistent_path_accepted(self, tmp_path: Path) -> None:
        result = normalize_hint_paths(["does/not/exist.py"], tmp_path)
        assert result == ["does/not/exist.py"]

    def test_duplicates_deduped_preserving_order(self, tmp_path: Path) -> None:
        result = normalize_hint_paths(["a.py", "b.py", "a.py", "c.py", "b.py"], tmp_path)
        assert result == ["a.py", "b.py", "c.py"]

    def test_empty_list_returns_empty(self, tmp_path: Path) -> None:
        assert normalize_hint_paths([], tmp_path) == []

    def test_absolute_and_relative_same_path_deduped(self, tmp_path: Path) -> None:
        result = normalize_hint_paths(["src/foo.py", str(tmp_path / "src" / "foo.py")], tmp_path)
        assert result == ["src/foo.py"]

    def test_absolute_with_embedded_dotdot_escaping_root_raises(self, tmp_path: Path) -> None:
        # An absolute path like /root/../../../etc/passwd still escapes after resolution
        escaping = str(tmp_path / ".." / ".." / "etc" / "passwd")
        with pytest.raises(ValueError, match="escapes"):
            normalize_hint_paths([escaping], tmp_path)

    def test_absolute_with_dotdot_staying_under_root_accepted(self, tmp_path: Path) -> None:
        # /root/sub/../other.py resolves to /root/other.py — still under root
        zigzag = str(tmp_path / "sub" / ".." / "other.py")
        result = normalize_hint_paths([zigzag], tmp_path)
        assert result == ["other.py"]

    def test_symlink_pointing_outside_root_rejected(self, tmp_path: Path) -> None:
        # resolve() follows symlinks; a hint through an in-repo symlink that points
        # outside the root escapes after resolution and must be rejected.
        inside = tmp_path / "link.py"
        outside = tmp_path.parent / "real.py"
        outside.write_text("x")
        inside.symlink_to(outside)
        with pytest.raises(ValueError, match="escapes"):
            normalize_hint_paths([str(inside)], tmp_path)

    def test_empty_string_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="empty"):
            normalize_hint_paths([""], tmp_path)

    def test_whitespace_only_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="empty"):
            normalize_hint_paths(["  "], tmp_path)

    def test_padded_path_trimmed_before_normalization(self, tmp_path: Path) -> None:
        # Padding must not persist into the stored hint, or it never matches the real file.
        result = normalize_hint_paths(["  src/foo.py  "], tmp_path)
        assert result == ["src/foo.py"]


class TestSlugify:
    def test_normalizes_text_and_trims_truncated_separator(self) -> None:
        assert slugify("Wire the Export Path!") == "wire-the-export-path"
        assert slugify("abc def", max_length=4) == "abc"

    def test_blank_and_punctuation_only_values_return_empty(self) -> None:
        assert slugify("  ") == ""
        assert slugify("!!!") == ""

    def test_truncation_collisions_are_not_disambiguated(self) -> None:
        first = slugify("abcdefghij first", max_length=10)
        second = slugify("abcdefghij second", max_length=10)
        assert first == second == "abcdefghij"


class TestResolveProjectPath:
    def test_valid_missing_path_is_resolved_under_root(self, tmp_path: Path) -> None:
        result = resolve_project_path("future/file.md", tmp_path, label="path")
        assert result == tmp_path / "future/file.md"

    def test_only_trusted_global_paths_may_escape_project_root(self, tmp_path: Path) -> None:
        outside = tmp_path.parent / "global.md"
        trusted = trust_global_path(str(outside), tmp_path)

        assert isinstance(trusted, TrustedGlobalPath)
        assert resolve_project_path(trusted, tmp_path, label="path") == outside
        with pytest.raises(ValueError, match="escapes"):
            resolve_project_path(str(outside), tmp_path, label="path")
