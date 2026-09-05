from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import typer
from typer.testing import CliRunner

from milknado.cli import app
from milknado.cli.run import watch as watch_command

runner = CliRunner()


def test_watch_command_requires_an_interactive_terminal(tmp_path: Path) -> None:
    with patch("milknado.cli.run._is_interactive_terminal", return_value=False):
        result = runner.invoke(app, ["watch", "--project-root", str(tmp_path)])

    assert result.exit_code == 2
    assert "interactive terminal" in result.output


def test_watch_command_launches_durable_observer(tmp_path: Path) -> None:
    db_path = tmp_path / ".milknado" / "milknado.db"
    config = SimpleNamespace(db_path=db_path)
    with (
        patch("milknado.cli.run._is_interactive_terminal", return_value=True),
        patch("milknado.cli.run._load_or_default", return_value=(config, [])),
        patch("milknado.app.watch_tui.run_watch_tui") as run_watch_tui,
    ):
        watch_command(tmp_path)

    run_watch_tui.assert_called_once_with(tmp_path.resolve(), db_path)


def test_watch_command_maps_missing_database_to_cli_error(tmp_path: Path) -> None:
    config = SimpleNamespace(db_path=tmp_path / "missing.db")
    with (
        patch("milknado.cli.run._is_interactive_terminal", return_value=True),
        patch("milknado.cli.run._load_or_default", return_value=(config, [])),
        patch(
            "milknado.app.watch_tui.run_watch_tui",
            side_effect=FileNotFoundError("missing database"),
        ),
        pytest.raises(typer.Exit) as exc,
    ):
        watch_command(tmp_path)

    assert exc.value.exit_code == 1
