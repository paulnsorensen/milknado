from __future__ import annotations

from starlette.applications import Starlette

from milknado.web.login import LaunchToken
from milknado.web.server import run_server


class Runner:
    def __init__(self) -> None:
        self.args = None

    def run(self, app, *, host, port) -> None:
        self.args = (app, host, port)


def test_server_binds_loopback_and_does_not_leak_token_to_opener(capsys) -> None:
    app = Starlette()
    login = LaunchToken("secret-token")
    runner = Runner()
    opened: list[str] = []
    run_server(app, login, port=8765, opener=opened.append, runner=runner)
    assert runner.args == (app, "127.0.0.1", 8765)
    assert opened == ["http://127.0.0.1:8765/"]
    assert "token=secret-token" in capsys.readouterr().out


def test_server_no_open_skips_opener() -> None:
    runner = Runner()
    opened: list[str] = []
    run_server(
        Starlette(), LaunchToken("token"), no_open=True, opener=opened.append, runner=runner
    )
    assert opened == []
