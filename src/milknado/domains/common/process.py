"""Process-liveness helper shared across slices (graph reclaim, dispatch orphan
recovery). A thin wrapper over `os.kill(pid, 0)` so no slice reaches into
another's internals for it.
"""

from __future__ import annotations

import os


def pid_alive(pid: int) -> bool:
    """True if a process with this pid exists on the local machine.

    `os.kill(pid, 0)` sends no signal but performs the existence + permission
    check. PermissionError means the process exists but is owned by another user
    (still alive); ProcessLookupError / other OSError means it is gone. Cross-machine
    runners are out of scope (the spec assumes runners are local to the daemon).
    """
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True
