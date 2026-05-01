from __future__ import annotations

import os
import re
import select
import subprocess
import sys
import time
from pathlib import Path

import pytest

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def _read_available(fd: int) -> str:
    chunks: list[bytes] = []
    while True:
        ready, _, _ = select.select([fd], [], [], 0.05)
        if not ready:
            break
        try:
            data = os.read(fd, 4096)
        except OSError:
            break
        if not data:
            break
        chunks.append(data)
    return b"".join(chunks).decode("utf-8", errors="replace")


def _wait_for_text(fd: int, needle: str, timeout_s: float) -> str:
    end = time.time() + timeout_s
    output = ""
    while time.time() < end:
        output += _read_available(fd)
        if needle in _strip_ansi(output):
            return output
        time.sleep(0.05)
    raise AssertionError(f"Timed out waiting for {needle!r}. Output so far:\n{output}")


def _milknado_db_rel(project_root: Path) -> Path:
    return (project_root / ".milknado" / "milknado.db").relative_to(project_root)


def _write_fake_planner(project_root: Path) -> Path:
    agent = project_root / "fake_planner.py"
    agent.write_text(
        (
            "import json\n"
            "import sys\n"
            "body = sys.stdin.read().lower()\n"
            'is_revision = "user revision request" in body\n'
            "payload = {\n"
            '  "manifest_version": "milknado.plan.v2",\n'
            '  "goal": "PTY plan goal",\n'
            '  "goal_summary": "Revised plan." if is_revision else "Initial plan.",\n'
            '  "spec_path": "spec.md",\n'
            '  "changes": [{"id": "c1", "path": "src/foo.py", "description": "Add foo"}],\n'
            '  "new_relationships": []\n'
            "}\n"
            'print("```json")\n'
            "print(json.dumps(payload))\n"
            'print("```")\n'
        ),
        encoding="utf-8",
    )
    return agent


def _start_plan_proc(project_root: Path, spec: Path) -> tuple[subprocess.Popen[str], int]:
    master_fd, slave_fd = os.openpty()
    try:
        env = dict(os.environ)
        src_root = str(Path(__file__).resolve().parents[1] / "src")
        env["PYTHONPATH"] = src_root + os.pathsep + env.get("PYTHONPATH", "")
        proc = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "from milknado.cli import app; app()",
                "plan",
                "--interactive",
                "--spec",
                str(spec),
                "--project-root",
                str(project_root),
            ],
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            cwd=project_root,
            env=env,
            close_fds=True,
        )
    finally:
        os.close(slave_fd)
    return proc, master_fd


@pytest.mark.skipif(sys.platform == "win32", reason="PTY smoke test requires POSIX")
def test_plan_interactive_pty_smoke(tmp_path: Path) -> None:
    project_root = tmp_path
    spec = project_root / "spec.md"
    spec.write_text("# PTY plan goal\n\nImplement something small.\n", encoding="utf-8")
    agent = _write_fake_planner(project_root)
    (project_root / "milknado.toml").write_text(
        (
            "[milknado]\n"
            'agent_family = "claude"\n'
            f'planning_agent = "{sys.executable} {agent}"\n'
            f'db_path = "{_milknado_db_rel(project_root)}"\n'
        ),
        encoding="utf-8",
    )
    proc, master_fd = _start_plan_proc(project_root, spec)

    try:
        output = _wait_for_text(master_fd, "Choose next step", timeout_s=10.0)
        os.write(master_fd, b"1\n")
        proc.wait(timeout=10.0)
        output += _read_available(master_fd)
    finally:
        os.close(master_fd)

    clean = _strip_ansi(output)
    assert proc.returncode == 0, output
    assert "Plan iteration 1" in clean
    assert "Choose next step" in clean


@pytest.mark.skipif(sys.platform == "win32", reason="PTY smoke test requires POSIX")
def test_plan_interactive_pty_revise_then_accept(tmp_path: Path) -> None:
    project_root = tmp_path
    spec = project_root / "spec.md"
    spec.write_text("# PTY plan goal\n\nImplement something small.\n", encoding="utf-8")
    agent = _write_fake_planner(project_root)
    (project_root / "milknado.toml").write_text(
        (
            "[milknado]\n"
            'agent_family = "claude"\n'
            f'planning_agent = "{sys.executable} {agent}"\n'
            f'db_path = "{_milknado_db_rel(project_root)}"\n'
        ),
        encoding="utf-8",
    )
    proc, master_fd = _start_plan_proc(project_root, spec)
    try:
        output = _wait_for_text(master_fd, "Choose next step", timeout_s=10.0)
        os.write(master_fd, b"2\n")
        output += _wait_for_text(master_fd, "What should change in the plan?", timeout_s=10.0)
        os.write(master_fd, b"Please include rollback notes.\n")
        output += _wait_for_text(master_fd, "Plan iteration 2", timeout_s=10.0)
        os.write(master_fd, b"1\n")
        proc.wait(timeout=10.0)
        output += _read_available(master_fd)
    finally:
        os.close(master_fd)

    clean = _strip_ansi(output)
    assert proc.returncode == 0, output
    assert "Plan iteration 1" in clean
    assert "Plan iteration 2" in clean
    assert "Choose next step" in clean
