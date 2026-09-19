# pyright: basic

from __future__ import annotations

import importlib
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from typing import cast

from milknado.cli.web import OwnerWebContext, OwnerWebOptions, OwnerWebServices
from milknado.domains.common import MilknadoConfig

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
        return SimpleNamespace(strict_exit=False)

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
    interrupts = iter((KeyboardInterrupt(), KeyboardInterrupt()))

    def interrupting_sleep(_seconds: float) -> None:
        raise next(interrupts)

    monkeypatch.setattr(web_module, "sleep", interrupting_sleep)

    server_active = Event()
    server_release = Event()

    def server(*args, **kwargs):
        server_active.set()
        server_release.wait()

    result = web_module.run_owner_web(
        OwnerWebContext(tmp_path, cast(MilknadoConfig, object()), []),
        OwnerWebOptions(),
        OwnerWebServices(server=server),
    )
    assert server_active.is_set()
    assert result is not None
    assert controller.stopped == 1
