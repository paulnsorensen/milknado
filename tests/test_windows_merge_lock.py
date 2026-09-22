from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(os.name != "nt", reason="requires Windows LockFileEx")


def _child(root: Path, events: Path, label: str, fail: bool = False) -> subprocess.Popen[str]:
    code = """
import sys
import time
from pathlib import Path
from milknado.domains.dispatch.isolate import _merge_back_lock
root = Path(sys.argv[1])
events = Path(sys.argv[2])
with _merge_back_lock(root):
    events.open("a", encoding="utf-8").write(f"start {sys.argv[3]} {time.monotonic()}\\n")
    if sys.argv[4] == "fail":
        raise RuntimeError("expected")
    time.sleep(0.2)
    events.open("a", encoding="utf-8").write(f"end {sys.argv[3]} {time.monotonic()}\\n")
"""
    return subprocess.Popen(
        [sys.executable, "-c", code, str(root), str(events), label, "fail" if fail else "ok"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def test_windows_merge_lock_serializes_children_and_releases_after_failure(tmp_path: Path) -> None:
    events = tmp_path / "events"
    first = _child(tmp_path, events, "first")
    second = _child(tmp_path, events, "second")
    assert first.wait(timeout=10) == 0
    assert second.wait(timeout=10) == 0
    values = [line.split() for line in events.read_text(encoding="utf-8").splitlines()]
    starts = {parts[1]: float(parts[2]) for parts in values if parts[0] == "start"}
    ends = {parts[1]: float(parts[2]) for parts in values if parts[0] == "end"}
    intervals = sorted((starts[label], ends[label]) for label in starts)
    assert intervals[0][1] <= intervals[1][0]

    failed = _child(tmp_path, events, "failed", fail=True)
    assert failed.wait(timeout=10) != 0
    recovered = _child(tmp_path, events, "recovered")
    assert recovered.wait(timeout=10) == 0


def test_windows_merge_lock_releases_after_abrupt_child_exit(tmp_path: Path) -> None:
    """Allow a new process to enter after TerminateProcess kills the lock holder."""
    events = tmp_path / "events"
    holder = _child(tmp_path, events, "holder")
    for _ in range(100):
        if events.exists() and events.read_text(encoding="utf-8").strip():
            break
        time.sleep(0.05)
    else:
        holder.kill()
        pytest.fail("lock holder did not enter")
    holder.terminate()
    assert holder.wait(timeout=10) != 0
    recovered = _child(tmp_path, events, "recovered")
    assert recovered.wait(timeout=10) == 0
