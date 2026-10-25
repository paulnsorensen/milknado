"""Fixed graph bootstrap for one worker lifeline."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import psutil

from milknado.domains.common import HelperIdentity
from milknado.domains.graph import WorkerEvidenceStore
from milknado.loop._lifeline import run_lifeline


def main() -> int:
    if len(sys.argv) != 5:
        return 2
    db_path = Path(sys.argv[1])
    try:
        read_fd = int(sys.argv[2])
        generation = int(sys.argv[4])
    except ValueError:
        return 2
    if read_fd < 0 or generation < 0 or not sys.argv[3]:
        return 2
    helper = HelperIdentity(sys.argv[3], generation, os.getpid(), psutil.Process().create_time())
    try:
        with WorkerEvidenceStore(db_path) as store:
            return run_lifeline(read_fd, helper, store)
    finally:
        os.close(read_fd)


if __name__ == "__main__":
    raise SystemExit(main())
