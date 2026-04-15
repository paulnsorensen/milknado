from __future__ import annotations

import re
import subprocess
from pathlib import Path

from milknado.domains.common.types import RebaseResult

_CONFLICT_FILE_RE = re.compile(
    r"^CONFLICT \(.*?\): (?:Merge conflict in |.*? -> )(.+)$",
    re.MULTILINE,
)


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
            subprocess.run(
                ["git", "rebase", "--abort"],
                cwd=worktree,
                capture_output=True,
                text=True,
            )
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
