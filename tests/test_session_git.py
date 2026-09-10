from __future__ import annotations

import os
import signal
import subprocess
import sys
from contextlib import suppress
from pathlib import Path

import pytest

from milknado.adapters import GitAdapter
from milknado.domains.common import GitOperationError, SessionContext
from tests._session_git_helpers import install_git_producer


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    ).stdout


def _kill_process_group(child: subprocess.Popen[str]) -> None:
    with suppress(ProcessLookupError):
        os.killpg(child.pid, signal.SIGKILL)


def _run_isolated(root: Path, source: str) -> tuple[str, str]:
    project = Path(__file__).parents[1]
    environment = {**os.environ, "PYTHONPATH": str(project)}
    child = subprocess.Popen(
        [sys.executable, "-c", source, str(root)],
        cwd=root,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        try:
            stdout, stderr = child.communicate(timeout=3)
        except subprocess.TimeoutExpired as exc:
            _kill_process_group(child)
            stdout, stderr = child.communicate()
            raise AssertionError(
                f"isolated Git inspection did not finish: stdout={stdout!r} stderr={stderr!r}"
            ) from exc
    finally:
        if child.poll() is None:
            _kill_process_group(child)
            _ = child.wait()
    if child.returncode != 0:
        raise AssertionError(f"isolated Git inspection failed: {stderr}")
    return stdout, stderr


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    _ = _git(tmp_path, "init", "-q")
    _ = _git(tmp_path, "config", "user.email", "test@example.com")
    _ = _git(tmp_path, "config", "user.name", "Test")
    _ = (tmp_path / "tracked.txt").write_text("before\n")
    _ = _git(tmp_path, "add", "tracked.txt")
    _ = _git(tmp_path, "commit", "-qm", "base")
    return tmp_path


def test_session_changes_include_committed_dirty_and_untracked_paths(repo: Path) -> None:
    base = _git(repo, "rev-parse", "HEAD").strip()
    _ = (repo / "tracked.txt").write_text("after\n")
    _ = (repo / "untracked.txt").write_text("new\n")
    _ = (repo / "committed.txt").write_text("committed\n")
    _ = _git(repo, "add", "committed.txt", "tracked.txt")
    _ = _git(repo, "commit", "-qm", "worker change")
    _ = (repo / "tracked.txt").write_text("dirty\n")
    context = SessionContext(family="omp", cwd=str(repo), base_oid=base)

    changes = GitAdapter(repo).session_changes(context)
    by_path = {change.path: change for change in changes}
    assert set(by_path) == {"committed.txt", "tracked.txt", "untracked.txt"}
    assert by_path["untracked.txt"].status == "??"

    adapter = GitAdapter(repo)
    assert "committed" in adapter.session_diff(context, "committed.txt")
    assert "dirty" in adapter.session_diff(context, "tracked.txt")
    assert "new" in adapter.session_diff(context, "untracked.txt")


def test_session_diff_rejects_paths_not_in_current_change_list(repo: Path) -> None:
    base = _git(repo, "rev-parse", "HEAD").strip()
    context = SessionContext(family="codex", cwd=str(repo), base_oid=base)

    with pytest.raises(ValueError, match="not in the session change list"):
        _ = GitAdapter(repo).session_diff(context, "missing.txt")


def test_session_diff_reports_binary_and_bounds_oversized_output(repo: Path) -> None:
    base = _git(repo, "rev-parse", "HEAD").strip()
    _ = (repo / "binary.dat").write_bytes(b"before\x00\n")
    _ = _git(repo, "add", "binary.dat")
    _ = _git(repo, "commit", "-qm", "binary base")
    _ = (repo / "binary.dat").write_bytes(b"after\x00\n")
    context = SessionContext(family="claude", cwd=str(repo), base_oid=base)
    binary = GitAdapter(repo).session_diff(context, "binary.dat")
    assert "Binary file" in binary

    _ = (repo / "large.txt").write_text("line\n" * 40_000)
    context = SessionContext(family="claude", cwd=str(repo), base_oid=base)
    large = GitAdapter(repo).session_diff(context, "large.txt")
    assert "truncated" in large
    assert len(large.encode()) <= 128 * 1024


def test_session_diff_stops_large_stdout_producer(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = _git(repo, "rev-parse", "HEAD").strip()
    _ = (repo / "tracked.txt").write_text("after\n")
    marker = repo / "stdout-complete"
    install_git_producer(repo, marker, 1)
    monkeypatch.setenv("PATH", f"{repo}{os.pathsep}{os.environ.get('PATH', '')}")
    context = SessionContext(family="omp", cwd=str(repo), base_oid=base)

    diff = GitAdapter(repo).session_diff(context, "tracked.txt")

    assert not marker.exists()
    assert "[diff truncated: file is too large to display]" in diff
    assert len(diff.encode()) <= 128 * 1024


def test_session_diff_stops_large_stderr_producer(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = _git(repo, "rev-parse", "HEAD").strip()
    _ = (repo / "tracked.txt").write_text("after\n")
    marker = repo / "stderr-complete"
    install_git_producer(repo, marker, 2)
    monkeypatch.setenv("PATH", f"{repo}{os.pathsep}{os.environ.get('PATH', '')}")
    context = SessionContext(family="omp", cwd=str(repo), base_oid=base)

    with pytest.raises(GitOperationError) as error:
        _ = GitAdapter(repo).session_diff(context, "tracked.txt")

    assert not marker.exists()
    assert len(error.value.detail.encode()) <= 128 * 1024


def test_session_changes_rejects_truncated_enumeration(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = _git(repo, "rev-parse", "HEAD").strip()
    marker = repo / "enumeration-complete"
    install_git_producer(repo, marker, 1, enumeration=True)
    monkeypatch.setenv("PATH", f"{repo}{os.pathsep}{os.environ.get('PATH', '')}")
    context = SessionContext(family="omp", cwd=str(repo), base_oid=base)

    with pytest.raises(GitOperationError, match="output truncated before complete enumeration"):
        _ = GitAdapter(repo).session_changes(context)

    assert not marker.exists()


def test_session_changes_reports_missing_base_and_worktree() -> None:
    adapter = GitAdapter(Path("/tmp/does-not-exist-milknado-session"))
    missing = SessionContext(family="omp", cwd="/tmp/does-not-exist-milknado-session")
    with pytest.raises(GitOperationError, match="worktree is unavailable"):
        _ = adapter.session_changes(missing)


def test_session_changes_preserves_rename_delete_binary_and_untracked_records(
    repo: Path,
) -> None:
    _ = (repo / "rename-source.txt").write_text("keep\nline\n")
    _ = (repo / "deleted.txt").write_text("gone\n")
    _ = (repo / "binary.dat").write_bytes(b"before\x00\n")
    _ = _git(repo, "add", "rename-source.txt", "deleted.txt", "binary.dat")
    _ = _git(repo, "commit", "-qm", "session base")
    _ = _git(repo, "config", "diff.renames", "true")
    base = _git(repo, "rev-parse", "HEAD").strip()

    _ = _git(repo, "mv", "rename-source.txt", "rename-target.txt")
    _ = (repo / "rename-target.txt").write_text("keep\nline\nmoved\n")
    (repo / "deleted.txt").unlink()
    _ = (repo / "binary.dat").write_bytes(b"after\x00\n")
    _ = (repo / "untracked.txt").write_text("new\n")
    context = SessionContext(family="omp", cwd=str(repo), base_oid=base)

    changes = {change.path: change for change in GitAdapter(repo).session_changes(context)}
    assert set(changes) == {
        "binary.dat",
        "deleted.txt",
        "rename-target.txt",
        "untracked.txt",
    }
    assert changes["rename-target.txt"].status == "R"
    assert changes["rename-target.txt"].old_path == "rename-source.txt"
    assert (changes["rename-target.txt"].added, changes["rename-target.txt"].removed) == (
        1,
        0,
    )
    assert changes["deleted.txt"].status == "D"
    assert (changes["deleted.txt"].added, changes["deleted.txt"].removed) == (0, 1)
    assert changes["binary.dat"].status == "M"
    assert (changes["binary.dat"].added, changes["binary.dat"].removed) == (None, None)
    assert changes["untracked.txt"].status == "??"
    assert (changes["untracked.txt"].added, changes["untracked.txt"].removed) == (1, 0)

    adapter = GitAdapter(repo)
    rename_diff = adapter.session_diff(context, "rename-target.txt")
    assert "rename from rename-source.txt" in rename_diff
    assert "rename to rename-target.txt" in rename_diff
    assert "-gone" in adapter.session_diff(context, "deleted.txt")
    assert adapter.session_diff(context, "binary.dat") == (
        "Binary file: binary.dat (unified diff unavailable)."
    )
    assert "+new" in adapter.session_diff(context, "untracked.txt")


def test_session_changes_rejects_untracked_symlink_escape(repo: Path) -> None:
    outside = repo.parent / "outside.txt"
    _ = outside.write_text("secret\n")
    (repo / "linked.txt").symlink_to(outside)
    base = _git(repo, "rev-parse", "HEAD").strip()
    context = SessionContext(family="omp", cwd=str(repo), base_oid=base)

    with pytest.raises(
        GitOperationError,
        match="git session changes failed: untracked path escapes worktree: linked.txt",
    ):
        _ = GitAdapter(repo).session_changes(context)


def test_session_diff_reports_untracked_special_paths_without_following(repo: Path) -> None:
    os.mkfifo(repo / "fifo")
    (repo / "fifo-link").symlink_to("fifo")
    source = """
import sys
from pathlib import Path

from milknado.adapters import GitAdapter
from milknado.domains.common import SessionContext

root = Path(sys.argv[1])
context = SessionContext(family="omp", cwd=str(root), base_oid="HEAD")
adapter = GitAdapter(root)
assert "fifo" not in {change.path for change in adapter.session_changes(context)}
print(adapter.session_diff(context, "fifo-link"))
"""

    stdout, _ = _run_isolated(repo, source)

    assert "Symlink: fifo-link -> fifo" in stdout


def test_session_changes_rejects_non_directory_worktree(repo: Path) -> None:
    worktree = repo / "tracked.txt"
    context = SessionContext(family="omp", cwd=str(worktree), base_oid="HEAD")

    with pytest.raises(
        GitOperationError,
        match=f"git session changes failed: worktree is not a directory: {worktree}",
    ):
        _ = GitAdapter(repo).session_changes(context)


def test_session_changes_rejects_blank_base_oid(repo: Path) -> None:
    context = SessionContext(family="omp", cwd=str(repo), base_oid=" \t")

    with pytest.raises(
        GitOperationError,
        match="git session changes failed: session base commit is unavailable",
    ):
        _ = GitAdapter(repo).session_changes(context)


def test_session_changes_rejects_unknown_base_oid(repo: Path) -> None:
    context = SessionContext(family="omp", cwd=str(repo), base_oid="missing-base")

    with pytest.raises(GitOperationError, match=r"git rev-parse --verify .* failed"):
        _ = GitAdapter(repo).session_changes(context)


def test_session_diff_accepts_path_starting_with_dash(repo: Path) -> None:
    filename = "-leading.txt"
    _ = (repo / filename).write_text("before\n")
    _ = _git(repo, "add", "--", filename)
    _ = _git(repo, "commit", "-qm", "dash path base")
    base = _git(repo, "rev-parse", "HEAD").strip()
    _ = (repo / filename).write_text("after\n")
    context = SessionContext(family="codex", cwd=str(repo), base_oid=base)

    diff = GitAdapter(repo).session_diff(context, filename)
    assert f"--- a/{filename}" in diff
    assert "+after" in diff
