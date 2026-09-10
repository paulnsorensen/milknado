"""Immutable records and bounded, read-only Git helpers for session views."""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

from milknado.adapters._git_session import (
    RunProcess,
    bound_diff,
    run_bounded_process,
    untracked_counts,
    untracked_diff,
)
from milknado.domains.common import GitOperationError, SessionContext


@dataclass(frozen=True, slots=True)
class ChangedFile:
    """One path changed between a session base and its worktree."""

    path: str
    status: str
    added: int | None
    removed: int | None
    old_path: str | None = None


RunGit: TypeAlias = Callable[[list[str], Path | None], subprocess.CompletedProcess[str]]


def _bounded_enumeration(args: list[str], root: Path, operation: str) -> str:
    stdout, stderr, returncode, reason = run_bounded_process(
        subprocess.Popen,
        args,
        root,
        None,
    )
    if reason is not None:
        raise GitOperationError(
            operation, f"{reason} output truncated before complete enumeration"
        )
    if returncode != 0:
        detail = (stderr or stdout).strip() or f"exit status {returncode}"
        raise GitOperationError(operation, detail)
    return stdout


def _bounded_diff_enumeration(root: Path, base: str, option: str, operation: str) -> str:
    return _bounded_enumeration(
        ["git", "diff", "--no-ext-diff", "--no-textconv", option, "-z", base, "--"],
        root,
        operation,
    )


def _session_root(context: SessionContext) -> Path:
    root = Path(context.cwd).expanduser()
    try:
        root = root.resolve(strict=True)
    except OSError as exc:
        raise GitOperationError("session changes", f"worktree is unavailable: {root}") from exc
    if not root.is_dir():
        raise GitOperationError("session changes", f"worktree is not a directory: {root}")
    return root


def _session_base(run_git: RunGit, root: Path, context: SessionContext) -> str:
    base = context.base_oid.strip()
    if not base:
        raise GitOperationError("session changes", "session base commit is unavailable")
    result = run_git(["rev-parse", "--verify", f"{base}^{{commit}}"], root)
    return result.stdout.strip() or base


def _parse_name_status(output: str) -> tuple[tuple[str, str, str | None], ...]:
    tokens = [token for token in output.split("\0") if token]
    changes: list[tuple[str, str, str | None]] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if "\t" in token:
            status, path = token.split("\t", 1)
            index += 1
        else:
            status = token
            index += 1
            if index >= len(tokens):
                break
            path = tokens[index]
            index += 1
        old_path: str | None = None
        if status[:1] in {"R", "C"}:
            old_path = path
            if index >= len(tokens):
                break
            path = tokens[index]
            index += 1
        changes.append((path, status[:1] or "M", old_path))
    return tuple(changes)


def _parse_numstat(output: str) -> dict[str, tuple[int | None, int | None]]:
    tokens = iter(output.split("\0"))
    stats: dict[str, tuple[int | None, int | None]] = {}
    for token in tokens:
        fields = token.split("\t", 2)
        if len(fields) != 3:
            continue
        added, removed, path = fields
        if not path:
            _ = next(tokens)
            path = next(tokens)
        if added == "-" or removed == "-":
            counts = (None, None)
        else:
            try:
                counts = (int(added), int(removed))
            except ValueError:
                continue
        stats[path] = counts
    return stats


def session_changes(run_git: RunGit, context: SessionContext) -> tuple[ChangedFile, ...]:
    """List base-to-worktree changes without touching the index."""
    root = _session_root(context)
    base = _session_base(run_git, root, context)
    names = _bounded_diff_enumeration(root, base, "--name-status", "diff --name-status")
    stats = _parse_numstat(_bounded_diff_enumeration(root, base, "--numstat", "diff --numstat"))
    changes = [
        ChangedFile(path, status, *stats.get(path, (None, None)), old_path)
        for path, status, old_path in _parse_name_status(names)
    ]
    tracked = {change.path for change in changes}
    untracked = _bounded_enumeration(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        root,
        "ls-files",
    ).split("\0")
    for path in untracked:
        if path and path not in tracked:
            changes.append(ChangedFile(path, "??", *untracked_counts(root, path)))
    return tuple(sorted(changes, key=lambda change: change.path))


def session_diff(
    run_git: RunGit,
    context: SessionContext,
    path: str,
    *,
    process_run: RunProcess = subprocess.Popen,
) -> str:
    root = _session_root(context)
    changes = session_changes(run_git, context)
    change = next((item for item in changes if item.path == path), None)
    if change is None:
        raise ValueError(f"path is not in the session change list: {path!r}")
    if change.status == "??":
        return untracked_diff(process_run, root, path)
    base = context.base_oid.strip()
    if not base:
        raise GitOperationError("session diff", "session base commit is unavailable")
    pathspecs = [path]
    if change.old_path is not None:
        pathspecs.insert(0, change.old_path)
    stdout, stderr, returncode, reason = run_bounded_process(
        process_run,
        ["git", "diff", "--no-ext-diff", "--no-textconv", base, "--", *pathspecs],
        root,
        None,
    )
    if reason == "stdout":
        return bound_diff(stdout, path, truncated=True)
    if reason == "stderr" or returncode != 0:
        raise GitOperationError("diff", (stderr or stdout).strip())
    return bound_diff(stdout, path)


def diff_for_review(
    run_git: RunGit,
    worktree: Path,
    base_oid: str,
    *,
    process_run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> str:
    try:
        tracked = run_git(
            ["diff", "--no-ext-diff", "--no-textconv", base_oid, "--"], worktree
        ).stdout
        untracked = run_git(
            ["ls-files", "--others", "--exclude-standard"], worktree
        ).stdout.splitlines()
        parts = [tracked]
        for path in untracked:
            result = process_run(
                [
                    "git",
                    "diff",
                    "--no-ext-diff",
                    "--no-textconv",
                    "--no-index",
                    "--",
                    "/dev/null",
                    path,
                ],
                cwd=worktree,
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode not in (0, 1):
                detail = (result.stderr or result.stdout).strip()
                raise GitOperationError("diff --no-index", detail)
            parts.append(result.stdout)
        return "\n".join(part.rstrip("\n") for part in parts if part)
    except OSError as exc:
        raise GitOperationError("diff for review", str(exc)) from exc
