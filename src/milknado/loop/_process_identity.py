"""Process identity checks and verified-target signaling."""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import time
from typing import Literal, TypeAlias

import psutil

from milknado.domains.common import WorkerIdentity

IdentityState: TypeAlias = Literal["live", "gone", "mismatch", "unknown"]
Descendant: TypeAlias = tuple[int, float, int]
_log = logging.getLogger(__name__)


def identity_state(pid: int, token: float) -> IdentityState:
    try:
        process = psutil.Process(pid)
        if process.create_time() != token:
            return "mismatch"
        return "gone" if process.status() == psutil.STATUS_ZOMBIE else "live"
    except psutil.NoSuchProcess:
        return "gone"
    except (psutil.AccessDenied, OSError):
        return "unknown"


def observe_descendants(worker: WorkerIdentity) -> tuple[Descendant, ...]:
    if identity_state(worker.pid, worker.start_token) != "live":
        return ()
    try:
        children = psutil.Process(worker.pid).children(recursive=True)
    except psutil.NoSuchProcess:
        return ()
    except (psutil.AccessDenied, OSError):
        raise RuntimeError("worker descendant enumeration unresolved") from None
    observed: list[Descendant] = []
    for child in children:
        try:
            if child.status() != psutil.STATUS_ZOMBIE:
                observed.append((child.pid, child.create_time(), os.getpgid(child.pid)))
        except psutil.NoSuchProcess:
            continue
        except (psutil.AccessDenied, OSError):
            raise RuntimeError("worker descendant enumeration unresolved") from None
    return tuple(observed)


def _group_state(pgid: int) -> IdentityState:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return "gone"
    except (PermissionError, OSError) as exc:
        _log.warning("worker group state unresolved pgid=%s errno=%s", pgid, exc.errno)
        return "unknown"
    return "live"


def _signal_identity(pid: int, token: float, signum: int) -> bool:
    if identity_state(pid, token) != "live":
        return False
    try:
        os.kill(pid, signum)
    except (ProcessLookupError, PermissionError, OSError):
        return False
    return True


def _signal_group(worker: WorkerIdentity, signum: int) -> bool:
    if identity_state(worker.pid, worker.start_token) != "live":
        return False
    try:
        if os.getpgid(worker.pid) != worker.pgid or worker.pgid != worker.pid:
            return False
        os.killpg(worker.pgid, signum)
    except (ProcessLookupError, PermissionError, OSError):
        return False
    return True


def _survivors(descendants: tuple[Descendant, ...]) -> tuple[Descendant, ...]:
    return tuple(
        target for target in descendants if identity_state(target[0], target[1]) == "live"
    )


def _signal_targets(
    worker: WorkerIdentity, descendants: tuple[Descendant, ...], signum: int
) -> None:
    _ = _signal_group(worker, signum)
    _ = _signal_identity(worker.pid, worker.start_token, signum)
    for pid, token, _pgid in descendants:
        _ = _signal_identity(pid, token, signum)


def _wait_targets(
    worker: WorkerIdentity, descendants: tuple[Descendant, ...], deadline: float
) -> None:
    while time.monotonic() < deadline:
        if identity_state(worker.pid, worker.start_token) != "live" and not _survivors(
            descendants
        ):
            return
        time.sleep(min(0.05, deadline - time.monotonic()))


def _confirm(
    worker: WorkerIdentity, descendants: tuple[Descendant, ...], deadline: float
) -> tuple[str, ...]:
    errors: set[str] = set()
    leader = identity_state(worker.pid, worker.start_token)
    if leader != "gone":
        errors.add(f"worker identity {leader}")
    for pid, token, _ in descendants:
        state = identity_state(pid, token)
        if state != "gone":
            errors.add(f"descendant {pid} {state}")
    group = _group_state(worker.pgid)
    while group != "gone" and time.monotonic() < deadline:
        time.sleep(min(0.05, deadline - time.monotonic()))
        group = _group_state(worker.pgid)
    if group != "gone":
        errors.add(f"worker group {group}")
    return tuple(sorted(errors))


def terminate_verified(
    worker: WorkerIdentity,
    retained: tuple[Descendant, ...],
    deadline: float,
    process: subprocess.Popen[bytes] | subprocess.Popen[str] | None = None,
) -> tuple[str, ...]:
    """Return unresolved covered targets; callers persist discovery before entry."""
    if os.name == "nt":
        return ("POSIX identity cleanup unavailable",)
    descendants = tuple(sorted(set(retained)))
    leader = identity_state(worker.pid, worker.start_token)
    errors = {f"worker identity {leader}"} if leader in ("mismatch", "unknown") else set()
    _signal_targets(worker, descendants, signal.SIGTERM)
    _wait_targets(worker, descendants, min(deadline, time.monotonic() + 0.5))
    if time.monotonic() < deadline:
        _signal_targets(worker, _survivors(descendants), signal.SIGKILL)
    _wait_targets(worker, descendants, deadline)
    if process is not None:
        try:
            _ = process.wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            errors.add("worker wait timed out")
    return tuple(sorted(errors | set(_confirm(worker, descendants, deadline))))
