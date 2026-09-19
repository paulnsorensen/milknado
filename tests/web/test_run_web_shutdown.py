# pyright: basic

from __future__ import annotations

import importlib
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from typing import cast

import pytest

from milknado.cli.web import OwnerWebContext, OwnerWebOptions, OwnerWebServices
from milknado.domains.common import MilknadoConfig

web_module = importlib.import_module("milknado.cli.web")


class Graph:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True

    def decide_goal_review(self, request: object, *, decided_by: str) -> object:
        return object()


class Controller:
    def __init__(self) -> None:
        self.stopped = 0
        self.called = Event()

    def run(self, **kwargs):
        self.called.set()
        return SimpleNamespace(strict_exit=False)

    def snapshot(self):
        return SimpleNamespace(active_runs=())

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

    def owner_commands_stub(controller, dependencies):
        assert dependencies.owner_capabilities is not None
        assert dependencies.owner_capabilities() is None
        return object()

    monkeypatch.setattr(web_module, "owner_commands", owner_commands_stub)
    interrupts = iter((KeyboardInterrupt(), KeyboardInterrupt()))

    def interrupting_sleep(_seconds: float) -> None:
        raise next(interrupts)

    monkeypatch.setattr(web_module, "sleep", interrupting_sleep)

    server_active = Event()
    server_release = Event()

    def server(*args, **kwargs):
        server_active.set()
        kwargs["options"].started()
        server_release.wait()

    result = web_module.run_owner_web(
        OwnerWebContext(tmp_path, cast(MilknadoConfig, object()), []),
        OwnerWebOptions(),
        OwnerWebServices(server=server),
    )
    assert server_active.is_set()
    assert result is not None
    assert controller.stopped == 1


def _configure(monkeypatch, graph: Graph, controller: Controller) -> None:
    monkeypatch.setattr(web_module, "ensure_db", lambda config, plugins: graph)
    monkeypatch.setattr(
        "milknado.app.run.build_execution_controller",
        lambda graph, config, root: controller,
    )
    monkeypatch.setattr("milknado.app.run.resolve_feature_branch", lambda root: "feature")
    monkeypatch.setattr(web_module, "create_app", lambda *args: object())
    monkeypatch.setattr(web_module, "owner_commands", lambda *args: object())


def test_owner_host_closes_graph_when_server_fails_before_bind(
    monkeypatch, tmp_path: Path
) -> None:
    graph = Graph()
    controller = Controller()
    _configure(monkeypatch, graph, controller)

    def server(*args, **kwargs):
        raise RuntimeError("bind failed")

    with pytest.raises(RuntimeError, match="bind failed"):
        web_module.run_owner_web(
            OwnerWebContext(tmp_path, cast(MilknadoConfig, object()), []),
            OwnerWebOptions(),
            OwnerWebServices(server=server),
        )
    assert graph.closed
    assert not controller.called.is_set()


def test_owner_host_stops_controller_when_server_fails_after_ready(
    monkeypatch, tmp_path: Path
) -> None:
    graph = Graph()
    controller = Controller()
    _configure(monkeypatch, graph, controller)

    def server(*args, **kwargs):
        kwargs["options"].started()
        controller.called.wait(timeout=1.0)
        raise RuntimeError("server failed")

    with pytest.raises(RuntimeError, match="server failed"):
        web_module.run_owner_web(
            OwnerWebContext(tmp_path, cast(MilknadoConfig, object()), []),
            OwnerWebOptions(),
            OwnerWebServices(server=server),
        )
    assert controller.called.is_set()
    assert controller.stopped == 1
    assert graph.closed


class BlockingController(Controller):
    def __init__(self) -> None:
        super().__init__()
        self.release = Event()

    def run(self, **kwargs):
        self.called.set()
        self.release.wait()
        return SimpleNamespace(strict_exit=False)


def test_owner_host_preserves_graph_when_controller_misses_shutdown_deadline(
    monkeypatch, tmp_path: Path
) -> None:
    graph = Graph()
    controller = BlockingController()
    _configure(monkeypatch, graph, controller)

    def server(*args, **kwargs):
        kwargs["options"].started()

    with pytest.raises(RuntimeError, match="shutdown deadline"):
        web_module.run_owner_web(
            OwnerWebContext(tmp_path, cast(MilknadoConfig, object()), []),
            OwnerWebOptions(),
            OwnerWebServices(server=server),
        )
    assert not graph.closed
    controller.release.set()


def test_owner_host_exits_on_first_interrupt_after_controller_completes(
    monkeypatch, tmp_path: Path
) -> None:
    graph = Graph()
    controller = Controller()
    _configure(monkeypatch, graph, controller)
    interrupts = iter((KeyboardInterrupt(),))

    def interrupting_sleep(_seconds: float) -> None:
        controller.called.wait(timeout=1.0)
        raise next(interrupts)

    monkeypatch.setattr(web_module, "sleep", interrupting_sleep)

    def server(*args, **kwargs):
        kwargs["options"].started()
        Event().wait()

    result = web_module.run_owner_web(
        OwnerWebContext(tmp_path, cast(MilknadoConfig, object()), []),
        OwnerWebOptions(),
        OwnerWebServices(server=server),
    )
    assert result is not None
    assert controller.stopped == 1
