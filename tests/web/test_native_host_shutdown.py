from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from pathlib import Path

import pytest

from milknado.cli._helpers import RunnableRootExclusions
from milknado.cli.web import OwnerWebContext, OwnerWebOptions, OwnerWebServices, run_owner_web, web
from milknado.domains.common import MilknadoConfig, PluginHook
from milknado.web import HostDependencies
from tests.web.test_run_web_shutdown import Controller, Graph

web_module = import_module("milknado.cli.web")


def _host_dependencies(shutdown: Callable[[], None]) -> Callable[..., HostDependencies]:
    def build(*_args: object) -> HostDependencies:
        return HostDependencies(shutdown=shutdown)

    return build


def _configure_host(monkeypatch: pytest.MonkeyPatch, graph: Graph, controller: Controller) -> None:
    def ensure(*_args: object) -> Graph:
        return graph

    def exclusions(*_args: object) -> RunnableRootExclusions:
        return RunnableRootExclusions(False, frozenset())

    def build_controller(*_args: object) -> Controller:
        return controller

    def feature(*_args: object) -> str:
        return "feature"

    def unused(*_args: object, **_kwargs: object) -> object:
        return object()

    monkeypatch.setattr(web_module, "ensure_db", ensure)
    monkeypatch.setattr("milknado.cli._helpers.apply_runnable_root_exclusions", exclusions)
    monkeypatch.setattr("milknado.app.run.build_execution_controller", build_controller)
    monkeypatch.setattr("milknado.app.run.resolve_feature_branch", feature)
    monkeypatch.setattr(web_module, "create_app", unused)
    monkeypatch.setattr(web_module, "owner_commands", unused)


def test_owner_host_awaits_native_shutdown_before_graph_close(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    graph = Graph()
    controller = Controller()
    _configure_host(monkeypatch, graph, controller)
    order: list[str] = []

    def shutdown() -> None:
        assert not graph.closed
        order.append("native")

    monkeypatch.setattr(web_module, "_host_dependencies", _host_dependencies(shutdown))

    def server(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("bind failed")

    with pytest.raises(RuntimeError, match="bind failed"):
        _ = run_owner_web(
            OwnerWebContext(tmp_path, MilknadoConfig(), []),
            OwnerWebOptions(),
            OwnerWebServices(server=server),
        )
    assert order == ["native"]
    assert graph.closed


def test_owner_host_preserves_graph_when_native_shutdown_is_uncertain(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    graph = Graph()
    controller = Controller()
    _configure_host(monkeypatch, graph, controller)

    def shutdown() -> None:
        assert not graph.closed
        raise RuntimeError("native worker shutdown is unconfirmed")

    monkeypatch.setattr(web_module, "_host_dependencies", _host_dependencies(shutdown))

    def server(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("bind failed")

    with pytest.raises(RuntimeError, match="native worker shutdown is unconfirmed"):
        _ = run_owner_web(
            OwnerWebContext(tmp_path, MilknadoConfig(), []),
            OwnerWebOptions(),
            OwnerWebServices(server=server),
        )
    assert not graph.closed


class _ObserverSource:
    def start(self) -> None:
        pass

    def close(self) -> None:
        pass


@pytest.fixture
def observer_host(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Graph, list[str]]:
    graph = Graph()
    config = MilknadoConfig(project_root=tmp_path, db_path=tmp_path / "graph.db")

    def load(_root: Path) -> tuple[MilknadoConfig, list[PluginHook]]:
        return config, []

    def ensure(*_args: object) -> Graph:
        return graph

    def watch_source(*_args: object) -> object:
        return object()

    def polled(_source: object) -> _ObserverSource:
        return _ObserverSource()

    def unused(*_args: object, **_kwargs: object) -> object:
        return object()

    def server(*_args: object) -> None:
        pass

    order: list[str] = []

    def shutdown() -> None:
        assert not graph.closed
        order.append("native")

    monkeypatch.setattr(graph, "register_controller_master", lambda: None, raising=False)
    monkeypatch.setattr(web_module, "load_or_default", load)
    monkeypatch.setattr(web_module, "ensure_db", ensure)
    monkeypatch.setattr(web_module, "_watch_source", watch_source)
    monkeypatch.setattr(web_module, "PolledSnapshotSource", polled)
    monkeypatch.setattr(web_module, "_host_dependencies", _host_dependencies(shutdown))
    monkeypatch.setattr(web_module, "observer_commands", unused)
    monkeypatch.setattr(web_module, "create_app", unused)
    monkeypatch.setattr(web_module, "run_server", server)
    return graph, order


def test_observer_host_awaits_native_shutdown_before_graph_close(
    observer_host: tuple[Graph, list[str]], tmp_path: Path
) -> None:
    graph, order = observer_host
    web(tmp_path, 8000, True)
    assert order == ["native"]
    assert graph.closed
