"""Inline dispatch applies the loop path's completion verifier (#498), in process."""

from __future__ import annotations

import subprocess
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from milknado.adapters import GitAdapter
from milknado.app.inline_gates import inline_completion_check
from milknado.domains.common import MikadoNode, MilknadoConfig, WorktreeMode
from milknado.domains.common.flavor_codec import Gate
from milknado.domains.common.flavor_profile import resolve_flavor_profile
from milknado.domains.execution import NO_GATES_CONFIGURED_MESSAGE
from milknado.mcp.run import (
    milknado_run_inline,
    milknado_run_inline_poll,
    milknado_run_inline_start,
)
from milknado.mcp.todo_mutate import milknado_todo_add

_TOML = '[milknado]\nquality_gates = ["{}"]\n'


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def _repo(root: Path, toml: str) -> None:
    root.mkdir(exist_ok=True)
    _ = _git(root, "init", "-b", "work")
    _ = _git(root, "config", "user.email", "t@t.com")
    _ = _git(root, "config", "user.name", "T")
    _ = (root / ".gitignore").write_text(".milknado/\nmilknado-*/\n")
    _ = (root / "milknado.toml").write_text(toml)
    _ = _git(root, "add", ".")
    _ = _git(root, "commit", "-m", "init")


def _check(
    root: Path, gates: tuple[Gate, ...] | None, mode: WorktreeMode
) -> Callable[[Path, str | None], str | None]:
    profile = resolve_flavor_profile(MilknadoConfig(quality_gates=gates), None)
    node = MikadoNode(id=1, description="d")
    return inline_completion_check(profile, node, GitAdapter(root), mode)


def test_check_without_configured_gates_fails_closed(tmp_path: Path) -> None:
    _repo(tmp_path, "")
    _ = (tmp_path / "f.txt").write_text("x")
    base = _git(tmp_path, "rev-parse", "HEAD")

    verdict = _check(tmp_path, None, WorktreeMode.ISOLATE)(tmp_path, base)

    assert verdict == NO_GATES_CONFIGURED_MESSAGE


def test_check_rejects_a_clean_worktree_even_when_gates_pass(tmp_path: Path) -> None:
    _repo(tmp_path, "")
    base = _git(tmp_path, "rev-parse", "HEAD")

    verdict = _check(tmp_path, (Gate(command="true"),), WorktreeMode.ISOLATE)(tmp_path, base)

    assert verdict is not None
    assert "no committed/stageable change" in verdict


def test_check_in_place_accepts_work_committed_since_dispatch(tmp_path: Path) -> None:
    _repo(tmp_path, "")
    check = _check(tmp_path, (Gate(command="true"),), WorktreeMode.THIS_BRANCH)
    _ = (tmp_path / "f.txt").write_text("x")
    _ = _git(tmp_path, "add", "f.txt")
    _ = _git(tmp_path, "commit", "-m", "worker")

    assert check(tmp_path, None) is None


def test_check_in_place_outside_a_repository_has_no_dispatch_base(tmp_path: Path) -> None:
    check = _check(tmp_path, (Gate(command="true"),), WorktreeMode.THIS_BRANCH)

    verdict = check(tmp_path, None)

    assert verdict is not None
    assert "git" in verdict


def _wait_terminal(run_id: str, root: Path) -> dict[str, object]:
    poll = getattr(milknado_run_inline_poll, "fn", milknado_run_inline_poll)
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        run = poll(run_id=run_id, project_root=str(root))
        if run["status"] != "running":
            return dict(run)
        time.sleep(0.1)
    raise AssertionError("run never finished")


@pytest.mark.parametrize("blocking", [True, False], ids=["inline", "start-poll"])
@pytest.mark.parametrize("case", [("exit 1", "failed"), ("true", "done")], ids=["fail", "pass"])
def test_inline_tools_run_the_gate_before_done(
    tmp_path: Path,
    worker_stub: Callable[[str], str],
    case: tuple[str, str],
    blocking: bool,
) -> None:
    gate, status = case
    root = tmp_path / "project"
    _repo(root, _TOML.format(gate))
    add = getattr(milknado_todo_add, "fn", milknado_todo_add)
    node = add(description="t", kind="task", project_root=str(root))
    tool = milknado_run_inline if blocking else milknado_run_inline_start
    run = getattr(tool, "fn", tool)(
        node_id=node["id"],
        worker_cmd=worker_stub("sh -c 'echo x > f.txt'"),
        timeout_seconds=20,
        project_root=str(root),
    )
    if not blocking:
        run = _wait_terminal(str(run["run_id"]), root)

    assert run["status"] == status, run
    assert (root / "f.txt").exists() is (status == "done")
    assert (run["error"] is not None) is (status == "failed")
