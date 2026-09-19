from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from typer.testing import CliRunner

from milknado.cli import app
from milknado.web.login import LaunchToken

runner = CliRunner()


def test_web_command_accepts_port_and_no_open(tmp_path: Path) -> None:
    token = LaunchToken("fixed")
    with (
        patch(
            "milknado.cli.web.load_or_default",
            return_value=(SimpleNamespace(db_path=tmp_path / "db"), []),
        ),
        patch("milknado.cli.web.ensure_db") as ensure_db,
        patch("milknado.cli.web.LaunchToken", return_value=token),
        patch("milknado.cli.web.PolledSnapshotSource") as polling,
        patch("milknado.cli.web.run_server") as server,
    ):
        ensure_db.return_value.close = lambda: None
        polling.return_value.start.return_value = None
        result = runner.invoke(
            app, ["web", "--project-root", str(tmp_path), "--port", "8123", "--no-open"]
        )
    assert result.exit_code == 0, result.output
    polling.return_value.start.assert_called_once()
    server.assert_called_once()
    assert server.call_args.kwargs == {"port": 8123, "no_open": True}


def test_run_web_options_delegate_to_owner_host(tmp_path: Path) -> None:
    with (
        patch("milknado.cli.run._load_or_default", return_value=(object(), [])),
        patch("milknado.cli.web.run_owner_web") as host,
    ):
        result = runner.invoke(
            app,
            ["run", "--web", "--no-open", "--port", "8124", "--project-root", str(tmp_path)],
        )
    assert result.exit_code == 0, result.output
    assert host.call_args.args[-3:] == (False, 8124, True)
