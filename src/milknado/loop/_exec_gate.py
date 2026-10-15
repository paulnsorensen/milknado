"""Gate the actual worker command until durable protection is ready."""

from __future__ import annotations

import os
import sys
from contextlib import suppress


def main() -> int:
    if len(sys.argv) < 4:
        return 72
    gate_fd, status_fd = map(int, sys.argv[1:3])
    try:
        release = os.read(gate_fd, 1)
    finally:
        os.close(gate_fd)
    if release != b"R":
        return 72
    command = sys.argv[3:]
    os.set_inheritable(status_fd, False)
    _ = os.write(status_fd, b"P")
    try:
        os.execvpe(command[0], command, os.environ)
    except OSError:
        with suppress(OSError):
            _ = os.write(status_fd, b"F")
        return 72
    except BaseException as exc:  # noqa: BLE001
        with suppress(OSError):
            _ = os.write(status_fd, b"X")
        print(f"worker exec gate error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 72
    finally:
        with suppress(OSError):
            os.close(status_fd)


if __name__ == "__main__":
    raise SystemExit(main())
