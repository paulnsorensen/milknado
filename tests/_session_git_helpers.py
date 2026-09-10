from __future__ import annotations

import shutil
import sys
from pathlib import Path


def install_git_producer(
    root: Path, marker: Path, stream: int, *, enumeration: bool = False
) -> None:
    real_git = shutil.which("git")
    assert real_git is not None
    fake_git = root / "git"
    _ = fake_git.write_text(
        f"""#!{sys.executable}
import os
import sys
from pathlib import Path

args = sys.argv[1:]
if {enumeration!r} and ("--name-status" in args or "--numstat" in args):
    chunk = chr(0x1D11E).encode() * 16384
    for _ in range(100):
        os.write({stream}, chunk)
    Path({str(marker)!r}).write_text("complete")
    raise SystemExit(0)
if "diff" in args and "--name-status" not in args and "--numstat" not in args:
    chunk = chr(0x1D11E).encode() * 16384
    for _ in range(100):
        os.write({stream}, chunk)
    Path({str(marker)!r}).write_text("complete")
    raise SystemExit(0)
os.execv({real_git!r}, [{real_git!r}, *sys.argv[1:]])
"""
    )
    _ = fake_git.chmod(0o755)
