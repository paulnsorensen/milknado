"""Process-liveness helper shared across slices (graph reclaim, dispatch orphan
recovery). A thin wrapper over `os.kill(pid, 0)` so no slice reaches into
another's internals for it.
"""

from __future__ import annotations

import os


def pid_alive(pid: int) -> bool:
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
