"""Process-liveness helper shared across slices (graph reclaim, dispatch orphan
recovery). A thin wrapper over `os.kill(pid, 0)` so no slice reaches into
another's internals for it.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class WorkerIdentity:
    invocation_id: str
    pid: int
    pgid: int
    start_token: float


@dataclass(frozen=True, slots=True)
class WorkerOwner:
    runtime_run_id: str
    supervisor_pid: int
    supervisor_start_token: float
    graph_run_id: str | None = None
    node_id: int | None = None


@dataclass(frozen=True, slots=True)
class HelperIdentity:
    invocation_id: str
    generation: int
    pid: int
    start_token: float


@dataclass(frozen=True, slots=True)
class ObservationKey:
    invocation_id: str
    owner: Literal["supervisor", "helper"]
    sequence: int
    generation: int
    pid: int
    start_token: float


CONTROLLER_MASTER_ENV = "MILKNADO_CONTROLLER_MASTER"
WORKER_CONTEXT_ENV = "MILKNADO_WORKER_CONTEXT"


def mark_worker_env(env: Mapping[str, str]) -> dict[str, str]:
    """Remove controller authority and mark a copy of the worker environment."""
    worker_env = dict(env)
    _ = worker_env.pop(CONTROLLER_MASTER_ENV, None)
    worker_env[WORKER_CONTEXT_ENV] = "1"
    return worker_env


def pid_alive(pid: object) -> bool:
    """True if a process with this pid exists on the local machine.

    `pid` is read from on-disk run state (an external boundary), so a malformed
    or non-positive value is treated as not alive rather than trusted: a non-int
    would raise `TypeError` and `pid=0` would target the *current process group*
    (making a wedged node look live forever), so both are rejected up front.

    `os.kill(pid, 0)` sends no signal but performs the existence + permission
    check. PermissionError means the process exists but is owned by another user
    (still alive); ProcessLookupError / other OSError means it is gone. Cross-machine
    runners are out of scope (the spec assumes runners are local to the daemon).
    """
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, OverflowError):
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True
