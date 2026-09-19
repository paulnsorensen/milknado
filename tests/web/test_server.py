# pyright: basic

from __future__ import annotations

import importlib
from threading import Event, Thread

import pytest
from starlette.applications import Starlette

from milknado.web.login import LaunchToken
from milknado.web.server import ServerOptions, ServerServices, run_server


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
    run_server(
        app,
        login,
        ServerOptions(port=8765),
        ServerServices(opener=opened.append, runner=runner),
    )
    assert runner.args == (app, "127.0.0.1", 8765)
    assert opened == ["http://127.0.0.1:8765/"]
    assert "token=secret-token" in capsys.readouterr().out


def test_server_no_open_skips_opener() -> None:
    runner = Runner()
    opened: list[str] = []
    run_server(
        Starlette(),
        LaunchToken("token"),
        ServerOptions(no_open=True),
        ServerServices(opener=opened.append, runner=runner),
    )
    assert opened == []


server_module = importlib.import_module("milknado.web.server")


class ServingServer:
    started = False
    should_exit = False

    def __init__(self) -> None:
        self.running = False
        self.release = Event()

    def run(self) -> None:
        self.started = True
        self.running = True
        self.release.wait()
        self.running = False


class FailingServer:
    started = False
    should_exit = True

    def run(self) -> None:
        raise OSError("bind failed")


def test_default_server_reports_ready_while_serving(monkeypatch) -> None:
    fake = ServingServer()
    monkeypatch.setattr(server_module.uvicorn, "Server", lambda config: fake)
    ready = Event()
    thread = Thread(
        target=run_server,
        args=(Starlette(), LaunchToken("token")),
        kwargs={"options": ServerOptions(no_open=True, started=ready.set)},
    )
    thread.start()
    assert ready.wait(timeout=1.0)
    assert fake.running
    fake.release.set()
    thread.join(timeout=1.0)
    assert not thread.is_alive()


def test_default_server_does_not_report_ready_on_bind_failure(monkeypatch) -> None:
    fake = FailingServer()
    monkeypatch.setattr(server_module.uvicorn, "Server", lambda config: fake)
    ready = Event()
    with pytest.raises(OSError, match="bind failed"):
        run_server(
            Starlette(),
            LaunchToken("token"),
            ServerOptions(no_open=True, started=ready.set),
        )
    assert not ready.is_set()
