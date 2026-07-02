"""The plugin SessionStart hook installs node-runner.js idempotently.

`workflows/` is not a recognized plugin component, so installing the milknado
plugin does not make the ultracode Workflow script discoverable. The hook at
plugins/milknado/hooks/hooks.json closes that gap: on SessionStart it copies
the bundled script into the project's .claude/workflows/ — copy when absent,
no-op when identical, refresh when the plugin copy changed, never delete.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_HOOKS_DIR = _REPO_ROOT / "plugins" / "milknado" / "hooks"
_HOOKS_JSON = _HOOKS_DIR / "hooks.json"
_INSTALL_SCRIPT = _HOOKS_DIR / "install-workflow.sh"
_RUNNER = "node-runner.js"


def _run_hook(plugin_root: Path | None, project_dir: Path) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if k != "CLAUDE_PLUGIN_ROOT"}
    if plugin_root is not None:
        env["CLAUDE_PLUGIN_ROOT"] = str(plugin_root)
    env["CLAUDE_PROJECT_DIR"] = str(project_dir)
    return subprocess.run(
        ["bash", str(_INSTALL_SCRIPT)],
        env=env,
        cwd=project_dir,
        capture_output=True,
        text=True,
    )


@pytest.fixture()
def plugin_root(tmp_path: Path) -> Path:
    root = tmp_path / "plugin"
    (root / "workflows").mkdir(parents=True)
    (root / "workflows" / _RUNNER).write_text("// v1\n", encoding="utf-8")
    return root


@pytest.fixture()
def project_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "project"
    directory.mkdir()
    return directory


def test_hooks_json_wires_session_start_to_install_script() -> None:
    config = json.loads(_HOOKS_JSON.read_text(encoding="utf-8"))
    matchers = config["hooks"]["SessionStart"]
    commands = [hook["command"] for matcher in matchers for hook in matcher["hooks"]]
    assert any(
        "${CLAUDE_PLUGIN_ROOT}/hooks/install-workflow.sh" in command for command in commands
    ), f"SessionStart must invoke the install script, got: {commands}"
    hook_types = {hook["type"] for matcher in matchers for hook in matcher["hooks"]}
    assert hook_types == {"command"}


def test_install_script_is_executable() -> None:
    assert os.access(_INSTALL_SCRIPT, os.X_OK), f"{_INSTALL_SCRIPT} must be executable"


def test_copies_when_absent(plugin_root: Path, project_dir: Path) -> None:
    result = _run_hook(plugin_root, project_dir)
    assert result.returncode == 0, result.stderr
    dest = project_dir / ".claude" / "workflows" / _RUNNER
    assert dest.read_text(encoding="utf-8") == "// v1\n"


def test_noops_when_identical(plugin_root: Path, project_dir: Path) -> None:
    dest = project_dir / ".claude" / "workflows" / _RUNNER
    dest.parent.mkdir(parents=True)
    dest.write_text("// v1\n", encoding="utf-8")
    mtime_before = dest.stat().st_mtime_ns

    result = _run_hook(plugin_root, project_dir)

    assert result.returncode == 0, result.stderr
    assert dest.stat().st_mtime_ns == mtime_before, "identical file must not be rewritten"
    assert dest.read_text(encoding="utf-8") == "// v1\n"


def test_refreshes_when_plugin_copy_changed(plugin_root: Path, project_dir: Path) -> None:
    dest = project_dir / ".claude" / "workflows" / _RUNNER
    dest.parent.mkdir(parents=True)
    dest.write_text("// v1\n", encoding="utf-8")
    (plugin_root / "workflows" / _RUNNER).write_text("// v2\n", encoding="utf-8")

    result = _run_hook(plugin_root, project_dir)

    assert result.returncode == 0, result.stderr
    assert dest.read_text(encoding="utf-8") == "// v2\n"


def test_exits_silently_when_plugin_root_unset(project_dir: Path) -> None:
    result = _run_hook(None, project_dir)
    assert result.returncode == 0
    assert result.stdout == ""
    assert result.stderr == ""
    assert not (project_dir / ".claude").exists()


def test_missing_source_never_deletes_existing_copy(plugin_root: Path, project_dir: Path) -> None:
    dest = project_dir / ".claude" / "workflows" / _RUNNER
    dest.parent.mkdir(parents=True)
    dest.write_text("// keep me\n", encoding="utf-8")
    (plugin_root / "workflows" / _RUNNER).unlink()

    result = _run_hook(plugin_root, project_dir)

    assert result.returncode == 0, result.stderr
    assert dest.read_text(encoding="utf-8") == "// keep me\n"
