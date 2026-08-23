"""Exit-code contract for scripts/check_dead_code.py: exercises the classifier at its
real CLI seam (subprocess, exit code + stdout), no mocking the classifier itself.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
GATE_SCRIPT = REPO_ROOT / "scripts" / "check_dead_code.py"


def _run_gate(*paths: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(GATE_SCRIPT), *paths],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_clean_tree_exits_zero() -> None:
    result = _run_gate("src")
    assert result.returncode == 0, result.stdout + result.stderr


def test_unclassified_dead_function_fails_the_gate(tmp_path: Path) -> None:
    probe = tmp_path / "probe.py"
    probe.write_text("def unused_probe():\n    return 1\n", encoding="utf-8")
    result = _run_gate(str(tmp_path))
    output = result.stdout + result.stderr
    assert result.returncode == 3
    assert "unused_probe" in output


def test_textual_lifecycle_override_is_classified_accepted(tmp_path: Path) -> None:
    probe = tmp_path / "probe.py"
    probe.write_text(
        textwrap.dedent(
            """\
            from textual.app import App, ComposeResult

            class ProbeApp(App):
                CSS = ""
                BINDINGS = []

                def compose(self) -> ComposeResult:
                    yield from ()

                def on_mount(self) -> None:
                    self.sub_title = "probe"

            ProbeApp()
            """
        ),
        encoding="utf-8",
    )
    result = _run_gate(str(tmp_path))
    output = result.stdout + result.stderr
    assert result.returncode == 0, output


def test_same_named_method_without_textual_base_is_not_classified(tmp_path: Path) -> None:
    probe = tmp_path / "probe.py"
    probe.write_text(
        textwrap.dedent(
            """\
            class NotTextual:
                def compose(self) -> None:
                    pass
            """
        ),
        encoding="utf-8",
    )
    result = _run_gate(str(tmp_path))
    output = result.stdout + result.stderr
    assert result.returncode == 3
    assert "compose" in output
