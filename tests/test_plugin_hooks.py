"""The plugin no longer auto-installs node-runner.js.

The former SessionStart hook (hooks/hooks.json -> hooks/install-workflow.sh)
copied node-runner.js into the project's .claude/workflows/ on every session
start. That auto-copy leaked into git in target repos that didn't ignore
.claude/workflows/, so it was removed by design. node-runner.js remains the
in-repo source-of-truth, installed manually or invoked by explicit scriptPath;
these tests pin that the auto-install is gone and that the runner keeps its
change-id -> node-id convention.
"""

from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PLUGIN_ROOT = _REPO_ROOT / "plugins" / "milknado"
_HOOKS_DIR = _PLUGIN_ROOT / "hooks"
_RUNNER = _PLUGIN_ROOT / "workflows" / "node-runner.js"


def test_session_start_auto_install_hook_is_removed() -> None:
    assert not (_HOOKS_DIR / "hooks.json").exists(), "SessionStart auto-install hook must be gone"
    assert not (_HOOKS_DIR / "install-workflow.sh").exists(), "install script must be gone"


def test_runner_header_documents_manual_install_not_hook() -> None:
    header = _RUNNER.read_text(encoding="utf-8")
    assert "install-workflow.sh" not in header, "runner header must not point at the removed hook"
    assert ".claude/workflows/" in header, "runner header must document the manual install path"


def test_change_id_convention_documented_in_header_and_wiki() -> None:
    """Locks the change-id -> node-id convention docs (spec acceptance 2 + 3)."""
    header = _RUNNER.read_text(encoding="utf-8")
    assert "id = str(node.id)" in header
    assert ".map(Number)" in header

    wiki = (
        _REPO_ROOT / ".hallouminate" / "wiki" / "history" / "workflow-executor-decision.md"
    ).read_text(encoding="utf-8")
    assert "str(node.id)" in wiki
    assert "map(Number)" in wiki
