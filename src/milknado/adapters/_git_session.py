"""Bounded Git process and untracked-path helpers for session views."""

from __future__ import annotations

import errno
import os
import selectors
import stat
import subprocess
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import IO, TypeAlias

from milknado.domains.common import GitOperationError

_MAX_DIFF_BYTES = 128 * 1024
_DIFF_TRUNCATION_MARKER = "\n\n[diff truncated: file is too large to display]\n"

RunProcess: TypeAlias = Callable[..., subprocess.Popen[bytes]]


def bound_diff(text: str, path: str, *, truncated: bool = False) -> str:
    if any(line.startswith("Binary files ") for line in text.splitlines()):
        return f"Binary file: {path} (unified diff unavailable)."
    encoded = text.encode("utf-8")
    if not truncated and len(encoded) <= _MAX_DIFF_BYTES:
        return text or f"No changes for {path}."
    marker = _DIFF_TRUNCATION_MARKER
    limit = max(0, _MAX_DIFF_BYTES - len(marker.encode("utf-8")))
    return encoded[:limit].decode("utf-8", errors="ignore") + marker


def _untracked_candidate(root: Path, path: str) -> Path:
    candidate = Path(os.path.normpath(root / path))
    if not candidate.is_relative_to(root):
        raise GitOperationError("session changes", f"untracked path escapes worktree: {path}")
    return candidate


def _untracked_kind(root: Path, path: str) -> tuple[Path, os.stat_result, str | None]:
    candidate = _untracked_candidate(root, path)
    try:
        info = candidate.lstat()
        target = str(candidate.readlink()) if stat.S_ISLNK(info.st_mode) else None
    except OSError as exc:
        raise GitOperationError("session changes", f"cannot read untracked path: {path}") from exc
    if target is not None:
        target_path = Path(os.path.normpath(candidate.parent / target))
        if not target_path.is_relative_to(root):
            raise GitOperationError("session changes", f"untracked path escapes worktree: {path}")
    return candidate, info, target


def _open_regular(candidate: Path) -> int | None:
    try:
        descriptor = os.open(candidate, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            return None
        raise
    try:
        regular = stat.S_ISREG(os.fstat(descriptor).st_mode)
    except OSError:
        os.close(descriptor)
        raise
    if not regular:
        os.close(descriptor)
        return None
    return descriptor


def untracked_counts(root: Path, path: str) -> tuple[int | None, int | None]:
    candidate, info, _ = _untracked_kind(root, path)
    if not stat.S_ISREG(info.st_mode):
        return (None, None)
    descriptor: int | None = None
    try:
        descriptor = _open_regular(candidate)
        if descriptor is None:
            return (None, None)
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = None
            data = stream.read(_MAX_DIFF_BYTES + 1)
    except OSError as exc:
        raise GitOperationError("session changes", f"cannot read untracked path: {path}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if len(data) > _MAX_DIFF_BYTES or b"\0" in data[:8192]:
        return (None, None)
    lines = data.count(b"\n")
    if data and not data.endswith(b"\n"):
        lines += 1
    return (lines, 0)


def _stop_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is None:
        with suppress(ProcessLookupError):
            _ = process.terminate()
        try:
            _ = process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            with suppress(ProcessLookupError):
                _ = process.kill()
            _ = process.wait()
    else:
        _ = process.wait()


def _capture_process(
    process: subprocess.Popen[bytes],
) -> tuple[bytes, bytes, str | None]:
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    streams: dict[int, tuple[str, IO[bytes]]] = {}
    selector = selectors.DefaultSelector()
    reason: str | None = None
    try:
        for name, stream in (("stdout", process.stdout), ("stderr", process.stderr)):
            if stream is not None:
                fd = stream.fileno()
                streams[fd] = (name, stream)
                _ = selector.register(fd, selectors.EVENT_READ)
        while selector.get_map() and reason is None:
            for key, _ in selector.select():
                fd = key.fd
                name, _ = streams[fd]
                capacity = _MAX_DIFF_BYTES + 1 - len(buffers[name])
                chunk = os.read(fd, min(64 * 1024, capacity))
                if not chunk:
                    _ = selector.unregister(fd)
                    continue
                buffers[name].extend(chunk)
                if len(buffers[name]) > _MAX_DIFF_BYTES:
                    reason = name
                    break
        if reason is None:
            _ = process.wait()
        else:
            _stop_process(process)
    except BaseException:
        _stop_process(process)
        raise
    finally:
        selector.close()
        for _, stream in streams.values():
            stream.close()
    return bytes(buffers["stdout"]), bytes(buffers["stderr"]), reason


def run_bounded_process(
    process_run: RunProcess,
    args: list[str],
    root: Path,
    stdin: int | None,
) -> tuple[str, str, int, str | None]:
    try:
        process = process_run(
            args,
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False,
            stdin=stdin,
        )
        stdout, stderr, reason = _capture_process(process)
    except OSError as exc:
        raise GitOperationError("diff", str(exc)) from exc
    return (
        stdout.decode("utf-8", errors="ignore"),
        stderr[:_MAX_DIFF_BYTES].decode("utf-8", errors="ignore"),
        process.returncode if process.returncode is not None else -1,
        reason,
    )


def _describe_untracked(path: str, info: os.stat_result, target: str | None) -> str | None:
    if target is not None:
        return f"Symlink: {path} -> {target}"
    if not stat.S_ISREG(info.st_mode):
        return f"Special file: {path} ({stat.filemode(info.st_mode)}; unified diff unavailable)."
    return None


def untracked_diff(process_run: RunProcess, root: Path, path: str) -> str:
    candidate, info, target = _untracked_kind(root, path)
    description = _describe_untracked(path, info, target)
    if description is not None:
        return description
    descriptor = _open_regular(candidate)
    if descriptor is None:
        return f"Special file: {path} (changed during inspection; unified diff unavailable)."
    # Stdin avoids Git treating Linux /dev/fd entries as symlinks.
    try:
        stdout, stderr, returncode, reason = run_bounded_process(
            process_run,
            [
                "git",
                "diff",
                "--no-ext-diff",
                "--no-textconv",
                "--no-index",
                "--",
                "/dev/null",
                "-",
            ],
            root,
            descriptor,
        )
    finally:
        os.close(descriptor)
    if info.st_mode & stat.S_IXUSR:
        stdout = stdout.replace("new file mode 100644\n", "new file mode 100755\n", 1)
    stdout = "".join(
        line.replace("/-", f"/{path}") if line.startswith(("diff --git ", "+++ ")) else line
        for line in stdout.splitlines(keepends=True)
    )
    if reason == "stdout":
        return bound_diff(stdout, path, truncated=True)
    if reason == "stderr" or returncode not in (0, 1):
        raise GitOperationError("diff --no-index", (stderr or stdout).strip())
    return bound_diff(stdout, path)
