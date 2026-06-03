"""Boundary behaviour of the shared `pid_alive` liveness helper.

`pid` reaches `pid_alive` from on-disk run state (an external boundary), so a
malformed or non-positive value must read as *not alive* rather than be trusted
to `os.kill`. The `pid=0` case is the one that matters most: `os.kill(0, 0)`
targets the caller's whole process group and returns success, which would make a
wedged node look perpetually live and never get reclaimed.
"""

import pytest

from milknado.domains.common import pid_alive


@pytest.mark.parametrize(
    "bad_pid",
    [
        0,  # os.kill(0, 0) hits the current process group — must NOT read as alive
        -1,  # negative pids address process groups, not a single process
        -12345,
        "1234",  # non-int from a corrupt JSON state file would raise TypeError in os.kill
        12.0,  # float likewise
        True,  # bool is an int subclass but is never a real pid
        None,
    ],
)
def test_pid_alive_rejects_invalid_pid(bad_pid):
    assert pid_alive(bad_pid) is False


def test_pid_alive_true_for_own_process():
    import os

    assert pid_alive(os.getpid()) is True
