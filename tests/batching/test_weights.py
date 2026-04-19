from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from milknado.domains.batching import FileChange, estimate_tokens


def test_delete_is_flat_80(tmp_path: Path) -> None:
    c = FileChange(id="1", path="x.py", edit_kind="delete")
    assert estimate_tokens(c, tmp_path) == 80


def test_rename_is_flat_120(tmp_path: Path) -> None:
    c = FileChange(id="1", path="x.py", edit_kind="rename")
    assert estimate_tokens(c, tmp_path) == 120


def test_modify_uses_tiktoken_on_existing_file(tmp_path: Path) -> None:
    f = tmp_path / "a.py"
    f.write_text("def hello():\n    return 42\n")
    c = FileChange(id="1", path="a.py", edit_kind="modify")
    with patch(
        "milknado.domains.batching.weights._tiktoken_count", return_value=10
    ):
        n = estimate_tokens(c, tmp_path)
    assert 5 < n < 100


def test_modify_falls_back_to_median_when_file_missing(tmp_path: Path) -> None:
    c = FileChange(id="1", path="missing.py", edit_kind="modify")
    # 150 lines * 10 tpl * 1.25 = 1875
    assert estimate_tokens(c, tmp_path) == 1875


def test_add_uses_median_lines_by_extension(tmp_path: Path) -> None:
    c = FileChange(id="1", path="new.py", edit_kind="add")
    assert estimate_tokens(c, tmp_path) == 1875  # 150 * 10 * 1.25


def test_unknown_extension_uses_defaults(tmp_path: Path) -> None:
    c = FileChange(id="1", path="new.xyz", edit_kind="add")
    assert estimate_tokens(c, tmp_path) == 1500  # 150 * 8 * 1.25


def test_rs_extension(tmp_path: Path) -> None:
    c = FileChange(id="1", path="lib.rs", edit_kind="add")
    # 200 * 11 * 1.25 = 2750
    assert estimate_tokens(c, tmp_path) == 2750
