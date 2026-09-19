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
    options = server.call_args.args[2]
    assert options.port == 8123
    assert options.no_open is True


def test_run_web_options_delegate_to_owner_host(tmp_path: Path) -> None:
    with (
        patch("milknado.cli.run._load_or_default", return_value=(object(), [])),
        patch("milknado.cli.web.run_owner_web") as host,
        patch("milknado.cli.run._print_run_result") as print_result,
    ):
        host.return_value = SimpleNamespace(strict_exit=False)
        result = runner.invoke(
            app,
            ["run", "--web", "--no-open", "--port", "8124", "--project-root", str(tmp_path)],
        )
    assert result.exit_code == 0, result.output
    context = host.call_args.args[0]
    assert context.project_root == tmp_path
    options = host.call_args.args[1]
    assert options.strict is False
    assert options.allow_protected is False
    assert options.port == 8124
    assert options.no_open is True
    print_result.assert_called_once_with(host.return_value)
