from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path

from milknado.domains.common.errors import RebaseAbortError
from milknado.domains.common.types import RebaseResult

_CONFLICT_FILE_RE = re.compile(
    r"^CONFLICT \(.*?\): (?:Merge conflict in |.*? -> )(.+)$",
    re.MULTILINE,
)

_logger = logging.getLogger(__name__)


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
        self._run(["worktree", "add", "-b", branch, str(path)])
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
        has_staged = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=worktree,
        ).returncode != 0
        if has_staged:
            self._run(["commit", "-m", msg], cwd=worktree)
