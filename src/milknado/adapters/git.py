from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
from pathlib import Path

from milknado.domains.common.errors import (
    GitOperationError,
    RebaseAbortError,
    UnlandedWorkError,
)
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
        try:
            return subprocess.run(
                ["git", *args],
                cwd=cwd or self._root,
                capture_output=True,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or str(exc)).strip()
            raise GitOperationError(" ".join(args), detail) from exc
        except OSError as exc:
            raise GitOperationError(" ".join(args), str(exc)) from exc

    def create_worktree(self, path: Path, branch: str) -> Path:
        self._run(["worktree", "add", "-b", branch, str(path)])
        return path

    def branch_exists(self, branch: str) -> bool:
        """True when a local branch named `branch` exists.

        Plumbing check with no side effects. `--quiet` makes the exit code the
        answer (0 = present, 1 = absent), so it must not go through `_run`,
        whose `check=True` would raise on the absent case.
        """
        result = subprocess.run(
            ["git", "show-ref", "--verify", "--quiet", f"refs/heads/{branch}"],
            cwd=self._root,
            capture_output=True,
            text=True,
        )
        return result.returncode == 0

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
        (unlanded): `merge-base --is-ancestor` ancestor containment; a
        non-zero or inconclusive probe refuses (fail closed).

        Raises UnlandedWorkError naming the worktree and the at-risk work.
        Raises ValueError when `path` is not itself a registered worktree —
        git commands run from a stale plain directory resolve against the
        PARENT repo, which would misreport the root's dirt as the worktree's;
        callers treat that as a non-refusal failure (nothing to preserve).
        """
        toplevel = self._run(["rev-parse", "--show-toplevel"], cwd=path).stdout.strip()
        if Path(toplevel).resolve() != path.resolve():
            raise ValueError(f"{path} is not a registered worktree (toplevel: {toplevel})")
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
        """True when `target_sha` already contains `head` (ancestor containment).

        Milknado's only land path is squash → rebase → `git merge --ff-only`,
        after which the landed HEAD equals the target tip, so `merge-base
        --is-ancestor` is exact. A non-zero or inconclusive probe returns False
        — refuse, don't guess.
        """
        ancestor = subprocess.run(
            ["git", "merge-base", "--is-ancestor", head, target_sha],
            cwd=worktree,
            capture_output=True,
            text=True,
        )
        return ancestor.returncode == 0

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

    def resolve_ref(self, ref: str) -> str:
        return self._run(["rev-parse", "--verify", ref]).stdout.strip()

    def compare_and_swap_ref(
        self,
        ref: str,
        expected_oid: str,
        new_oid: str,
    ) -> None:
        symbolic = subprocess.run(
            ["git", "symbolic-ref", "-q", "HEAD"],
            cwd=self._root,
            capture_output=True,
            text=True,
        )
        if symbolic.returncode not in (0, 1):
            raise GitOperationError("symbolic-ref -q HEAD", symbolic.stderr.strip())
        checked_out = symbolic.stdout.strip()
        normalized_ref = ref if ref.startswith("refs/") else f"refs/heads/{ref}"
        synchronize = checked_out == normalized_ref
        if synchronize:
            tracked = subprocess.run(
                ["git", "diff-index", "--quiet", expected_oid, "--"],
                cwd=self._root,
            )
            if tracked.returncode == 1:
                raise GitOperationError(
                    f"update-ref {ref}",
                    "checked-out target has tracked changes",
                )
            if tracked.returncode != 0:
                raise GitOperationError(
                    f"diff-index --quiet {expected_oid} --",
                    f"git exited {tracked.returncode}",
                )
            if self.resolve_ref("HEAD") != expected_oid:
                raise GitOperationError(
                    f"update-ref {ref}",
                    "checked-out target no longer matches expected OID",
                )
        self._run(["update-ref", normalized_ref, new_oid, expected_oid])
        if not synchronize:
            return
        try:
            self._run(["read-tree", "-u", "-m", expected_oid, new_oid])
        except GitOperationError:
            try:
                self._run(["read-tree", "-u", "-m", new_oid, expected_oid])
                self._run(["update-ref", normalized_ref, expected_oid, new_oid])
            except GitOperationError:
                _logger.exception(
                    "failed to roll back ref and worktree after synchronization failure"
                )
            raise

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
