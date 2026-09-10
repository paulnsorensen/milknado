from __future__ import annotations

from os import stat_result
from pathlib import Path

import pytest

from milknado.adapters import GitAdapter
from milknado.domains.common import GitOperationError, SessionContext
from tests.test_session_git import repo as repo


def test_internal_symlink_diff_does_not_disclose_target_contents(repo: Path) -> None:
    secret = "target contents must not appear in the link diff"
    _ = (repo / "target.txt").write_text(secret, encoding="utf-8")
    (repo / "link.txt").symlink_to("target.txt")
    adapter = GitAdapter(repo)
    context = SessionContext(family="omp", cwd=str(repo), base_oid="HEAD")

    changes = {change.path: change for change in adapter.session_changes(context)}
    assert (changes["link.txt"].added, changes["link.txt"].removed) == (None, None)
    assert (changes["target.txt"].added, changes["target.txt"].removed) == (1, 0)
    diff = adapter.session_diff(context, "link.txt")
    assert "link.txt" in diff and "target.txt" in diff
    assert secret not in diff


def test_disappearing_untracked_file_reports_inspection_failure(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = repo / "vanishing.txt"
    _ = candidate.write_text("pending change", encoding="utf-8")
    original_lstat = Path.lstat

    def remove_before_stat(path: Path) -> stat_result:
        if path == candidate:
            path.unlink()
        return original_lstat(path)

    monkeypatch.setattr(Path, "lstat", remove_before_stat)
    context = SessionContext(family="omp", cwd=str(repo), base_oid="HEAD")
    with pytest.raises(GitOperationError, match="cannot read untracked path: vanishing.txt"):
        _ = GitAdapter(repo).session_changes(context)
