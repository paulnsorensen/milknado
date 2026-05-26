from __future__ import annotations

import logging
import re
import shutil
import subprocess
from pathlib import Path

from milknado.domains.common.errors import RebaseAbortError
from milknado.domains.common.types import RebaseResult

_CONFLICT_FILE_RE = re.compile(
    r"^CONFLICT \(.*?\): (?:Merge conflict in |.*? -> )(.+)$",
    re.MULTILINE,
)

_logger = logging.getLogger(__name__)


def _try_mergiraf_resolve(worktree: Path, files: tuple[str, ...]) -> bool:
    """Attempt mergiraf solve on each conflicted file. Returns True if all succeed."""
    if shutil.which("mergiraf") is None:
        return False
    _logger.info("mergiraf: attempting to resolve %d conflicted file(s)", len(files))
    for file in files:
        rc = subprocess.run(
            ["mergiraf", "solve", str(worktree / file)],
            capture_output=True,
            text=True,
            timeout=30,
        ).returncode
        if rc != 0:
            _logger.info("mergiraf: failed on %s (rc=%d)", file, rc)
            return False
    _logger.info("mergiraf: all files resolved successfully")
    return True


class GitAdapter:
    def __init__(self, repo_root: Path) -> None:
        self._root = repo_root

    def _run(self, args: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args],
            cwd=cwd or self._root,
            capture_output=True,
            text=True,
            check=True,
        )

    def create_worktree(self, path: Path, branch: str) -> Path:
        # Auto-recover from an interrupted prior run. A leftover worktree dir or
        # a branch of the same name makes `git worktree add -b` fail fatally
        # (exit 255), which the run loop converts into a permanent node failure.
        # Best-effort drop a stale worktree at this path, prune the admin state,
        # then `add -B` to create-or-reset the branch from HEAD so re-dispatch is
        # idempotent (orphaned commits on the old branch are discarded).
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(path)],
            cwd=self._root,
            capture_output=True,
            text=True,
        )
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
        self._run(["worktree", "prune"])
        self._run(["worktree", "add", "-B", branch, str(path)])
        return path

    def remove_worktree(self, path: Path) -> None:
        self._run(["worktree", "remove", "--force", str(path)])

    def rebase(self, worktree: Path, onto: str) -> RebaseResult:
        result = subprocess.run(
            ["git", "rebase", onto],
            cwd=worktree,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            combined = result.stdout + result.stderr
            files = tuple(_CONFLICT_FILE_RE.findall(combined))
            if files and _try_mergiraf_resolve(worktree, files):
                add_result = subprocess.run(
                    ["git", "add", "-A"],
                    cwd=worktree,
                    capture_output=True,
                    text=True,
                )
                continue_result = subprocess.run(
                    ["git", "rebase", "--continue"],
                    cwd=worktree,
                    capture_output=True,
                    text=True,
                    env={**__import__("os").environ, "GIT_EDITOR": "true"},
                )
                if add_result.returncode == 0 and continue_result.returncode == 0:
                    _logger.info("mergiraf: rebase completed successfully after resolve")
                    return RebaseResult(success=True)
                _logger.info("mergiraf: rebase --continue failed, falling through to abort")
            abort_result = subprocess.run(
                ["git", "rebase", "--abort"],
                cwd=worktree,
                capture_output=True,
                text=True,
            )
            if abort_result.returncode != 0:
                _logger.error(
                    "git rebase --abort failed in %s: %s",
                    worktree,
                    abort_result.stderr,
                )
                raise RebaseAbortError(worktree, stderr=abort_result.stderr)
            return RebaseResult(
                success=False,
                conflicting_files=files,
                detail=combined.strip(),
            )
        return RebaseResult(success=True)

    def current_branch(self) -> str:
        result = self._run(["rev-parse", "--abbrev-ref", "HEAD"])
        return result.stdout.strip()

    def branch_exists(self, branch: str) -> bool:
        """Return True if a local branch with this name exists."""
        result = subprocess.run(
            ["git", "rev-parse", "--verify", f"refs/heads/{branch}"],
            cwd=self._root,
            capture_output=True,
            text=True,
        )
        return result.returncode == 0

    def is_ancestor(self, ref: str, of_ref: str) -> bool:
        """Return True if *ref* is reachable from *of_ref* (i.e. merged into it)."""
        result = subprocess.run(
            ["git", "merge-base", "--is-ancestor", ref, of_ref],
            cwd=self._root,
            capture_output=True,
            text=True,
        )
        return result.returncode == 0

    def push_branch(self, branch: str, remote: str = "origin") -> None:
        self._run(["push", remote, f"{branch}:{branch}"])

    def commit_all(self, worktree: Path, message: str) -> None:
        self._run(["add", "-A"], cwd=worktree)
        self._run(["commit", "-m", message], cwd=worktree)

    def squash_and_commit(self, worktree: Path, onto: str, msg: str) -> None:
        self._run(["add", "-A"], cwd=worktree)
        try:
            base_result = subprocess.run(
                ["git", "merge-base", "HEAD", onto],
                cwd=worktree,
                capture_output=True,
                text=True,
                check=True,
            )
            base = base_result.stdout.strip()
            self._run(["reset", "--soft", base], cwd=worktree)
        except subprocess.CalledProcessError:
            pass
        has_staged = (
            subprocess.run(
                ["git", "diff", "--cached", "--quiet"],
                cwd=worktree,
            ).returncode
            != 0
        )
        if has_staged:
            self._run(["commit", "-m", msg], cwd=worktree)
