from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
from pathlib import Path

from milknado.domains.common.errors import RebaseAbortError, UnlandedWorkError
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
        self._run(["worktree", "add", "-b", branch, str(path)])
        return path

    def remove_worktree(self, path: Path, target: str = "HEAD") -> None:
        """Fail-closed worktree removal: refuse dirty or unlanded worktrees.

        `target` is a ref resolved in the ROOT repo (default "HEAD" — the
        checked-out feature branch milknado lands work onto via fast_forward).
        The landed check runs against the worktree's HEAD *at call time*: by the
        time `rebase_and_merge`'s finally reaches here, squash + rebase have
        already rewritten the worktree history, so a fast-forwarded HEAD equals
        the target tip (trivially landed) while an aborted or never-merged HEAD
        correctly shows unlanded commits.

        Layer 1 (dirty): pre-check `git status --porcelain`, then remove without
        `--force` so git's native dirty guard stays as backstop. Layer 2
        (unlanded): `merge-base --is-ancestor` primary, `merge-tree` content
        equality fallback so squash/rebase-landed work is recognized; a
        conflicting or inconclusive probe refuses (fail closed).

        Raises UnlandedWorkError naming the worktree and the at-risk work.
        """
        dirty = self._run(["status", "--porcelain"], cwd=path).stdout.strip()
        if dirty:
            raise UnlandedWorkError(path, f"dirty files:\n{dirty}")
        target_sha = self._run(["rev-parse", target]).stdout.strip()
        head = self._run(["rev-parse", "HEAD"], cwd=path).stdout.strip()
        if not self._is_landed(path, head, target_sha):
            unlanded = self._run(
                ["log", "--oneline", f"{target_sha}..HEAD"], cwd=path
            ).stdout.strip()
            raise UnlandedWorkError(path, f"unlanded commits (not on {target}):\n{unlanded}")
        self._run(["worktree", "remove", str(path)])

    def _is_landed(self, worktree: Path, head: str, target_sha: str) -> bool:
        """True when `head` introduces nothing `target_sha` doesn't already have.

        Ancestor containment is exact for milknado's own land path (`git merge
        --ff-only` makes the landed HEAD equal the target tip). The merge-tree
        fallback recognizes externally squash/rebase-landed content (new SHAs,
        same tree); a conflicting probe returns False — refuse, don't guess.
        """
        ancestor = subprocess.run(
            ["git", "merge-base", "--is-ancestor", head, target_sha],
            cwd=worktree,
            capture_output=True,
            text=True,
        )
        if ancestor.returncode == 0:
            return True
        merged = subprocess.run(
            ["git", "merge-tree", "--write-tree", target_sha, head],
            cwd=worktree,
            capture_output=True,
            text=True,
        )
        if merged.returncode != 0:
            return False
        merged_tree = merged.stdout.strip().splitlines()[0] if merged.stdout.strip() else ""
        target_tree = self._run(["rev-parse", f"{target_sha}^{{tree}}"]).stdout.strip()
        return merged_tree == target_tree

    def force_remove_worktree(self, path: Path) -> None:
        """Explicitly destructive removal: discards dirty AND unlanded work.

        The only `--force` teardown in the codebase — reachable solely via
        `WorktreeManager.discard`, the named destructive path.
        """
        self._run(["worktree", "remove", "--force", str(path)])

    def prune_worktrees(self) -> None:
        self._run(["worktree", "prune"])

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
                if self._rebase_continue(worktree):
                    return RebaseResult(success=True)
            return self._abort_rebase(worktree, files, combined)
        return RebaseResult(success=True)

    def _rebase_continue(self, worktree: Path) -> bool:
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
            env={**os.environ, "GIT_EDITOR": "true"},
        )
        if add_result.returncode == 0 and continue_result.returncode == 0:
            _logger.info("mergiraf: rebase completed successfully after resolve")
            return True
        _logger.info("mergiraf: rebase --continue failed, falling through to abort")
        return False

    def _abort_rebase(self, worktree: Path, files: tuple[str, ...], combined: str) -> RebaseResult:
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

    def current_branch(self) -> str:
        result = self._run(["rev-parse", "--abbrev-ref", "HEAD"])
        return result.stdout.strip()

    def commit_all(self, worktree: Path, message: str) -> None:
        self._run(["add", "-A"], cwd=worktree)
        self._run(["commit", "-m", message], cwd=worktree)

    def squash_and_commit(self, worktree: Path, onto: str, msg: str) -> bool:
        """Squash the worktree's work into one commit on top of `onto`'s merge-base.

        Returns True when a commit was created, False when there was nothing to
        commit. The caller uses this to decide whether to fast-forward the feature
        branch — a no-commit squash has nothing to merge back.
        """
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
        return has_staged

    def fast_forward(self, branch: str) -> None:
        """Fast-forward the feature branch (checked out in the project root) to the
        rebased worker `branch` via `git merge --ff-only`.

        Run from the project root with the feature branch checked out, this advances
        it to the squashed-then-rebased worker commit. A non-ff merge — e.g. the
        project root's tree is dirty or the feature branch has diverged — fails loud:
        `git merge --ff-only` exits non-zero and `check=True` raises.
        """
        self._run(["merge", "--ff-only", branch])
