"""Gate the actual worker command until durable protection is ready."""

from __future__ import annotations

import os
import sys


def main() -> int:
    if len(sys.argv) < 4:
        return 72
    gate_fd = int(sys.argv[1])
    try:
        release = os.read(gate_fd, 1)
    finally:
        os.close(gate_fd)
    if release != b"R":
        return 72
    command = sys.argv[2:]
    os.execvpe(command[0], command, os.environ)
    return 72


if __name__ == "__main__":
    raise SystemExit(main())
