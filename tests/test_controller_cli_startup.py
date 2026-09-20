from __future__ import annotations

import importlib
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from milknado.cli import app
from milknado.domains.common import CONTROLLER_MASTER_ENV, default_config
from milknado.domains.graph import MikadoGraph

runner = CliRunner()


def _project_with_ready_nodes(project_root: Path) -> None:
    project_root.mkdir()
    _ = runner.invoke(app, ["init", str(project_root)])
    with (project_root / "milknado.toml").open("a", encoding="utf-8") as config:
        _ = config.write("\n[milknado.flavor.implement]\nreview = false\n")
    graph = MikadoGraph(default_config(project_root).db_path)
    root = graph.add_node("root goal")
    _ = graph.add_node("leaf task", parent_id=root.id)
    graph.close()


def test_interactive_run_without_secret_enters_tui_after_controller_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_root = tmp_path / "project"
    _project_with_ready_nodes(project_root)
    monkeypatch.delenv(CONTROLLER_MASTER_ENV, raising=False)
    cli_run = importlib.import_module("milknado.cli.run")
    monkeypatch.setattr(cli_run, "_is_interactive_terminal", lambda: True)
    tui = MagicMock(return_value=None)
    monkeypatch.setattr("milknado.app.run_tui.run_execution_tui", tui)
    with (
        patch("milknado.adapters.LoopAdapter"),
        patch("milknado.adapters.GitAdapter") as git,
        patch("milknado.adapters.CrgAdapter"),
    ):
        git.return_value.current_branch.return_value = "feature-x"  # pyright: ignore[reportAny]
        result = runner.invoke(app, ["run", "--project-root", str(project_root)])

    assert result.exit_code == 0, result.output
    tui.assert_called_once()
    with sqlite3.connect(default_config(project_root).db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM controller_master").fetchone() == (1,)


def test_watch_modes_do_not_load_credentials_when_store_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_root = tmp_path / "project"
    _project_with_ready_nodes(project_root)
    cli_run = importlib.import_module("milknado.cli.run")
    monkeypatch.setattr(cli_run, "_is_interactive_terminal", lambda: True)
    monkeypatch.setenv("XDG_STATE_HOME", "relative-state")
    normal = MagicMock()
    attached = MagicMock()
    monkeypatch.setattr("milknado.app.watch_tui.run_watch_tui", normal)
    monkeypatch.setattr("milknado.app.watch_tui.run_attached_watch_tui", attached)

    readonly = runner.invoke(app, ["watch", "--project-root", str(project_root)])
    owner = runner.invoke(app, ["watch", "--attached", "--project-root", str(project_root)])

    assert readonly.exit_code == 0, readonly.output
    assert owner.exit_code == 0, owner.output
    normal.assert_called_once()
    attached.assert_called_once()
