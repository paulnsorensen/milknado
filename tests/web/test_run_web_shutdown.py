from __future__ import annotations

import importlib
from pathlib import Path
from threading import Event

web_module = importlib.import_module("milknado.cli.web")


class Graph:
    def close(self) -> None:
        pass


class Controller:
    def __init__(self) -> None:
        self.stopped = 0
        self.called = Event()

    def run(self, **kwargs):
        self.called.set()
        raise KeyboardInterrupt

    def stop_scheduling(self) -> None:
        self.stopped += 1


def test_owner_host_stops_scheduling_on_first_interrupt(monkeypatch, tmp_path: Path) -> None:
    controller = Controller()
    graph = Graph()
    monkeypatch.setattr(web_module, "ensure_db", lambda config, plugins: graph)
    monkeypatch.setattr(
        "milknado.app.run.build_execution_controller",
        lambda graph, config, root: controller,
    )
    monkeypatch.setattr("milknado.app.run.resolve_feature_branch", lambda root: "feature")
    monkeypatch.setattr(web_module, "create_app", lambda *args: object())
    monkeypatch.setattr(web_module, "owner_commands", lambda *args: object())
    monkeypatch.setattr(
        web_module, "sleep", lambda seconds: (_ for _ in ()).throw(KeyboardInterrupt)
    )

    server_started = Event()

    def server(*args, **kwargs):
        server_started.set()

    web_module.run_owner_web(tmp_path, object(), [], False, False, 8000, True, server=server)
    assert server_started.is_set()
    assert controller.stopped == 1
