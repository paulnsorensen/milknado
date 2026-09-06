from __future__ import annotations

import os
from collections.abc import Callable, Generator
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from milknado.domains.common.protocols import CrgPort, GitPort, LoopPort
from milknado.domains.graph import MikadoGraph
from tests.worker_fixtures import install_worker_stub

pytest_plugins = ("tests.rebalance_helpers",)


@pytest.fixture(autouse=True)
def _isolate_global_milknado_config(  # pyright: ignore[reportUnusedFunction]
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Point XDG_CONFIG_HOME at an empty tmp dir so the dev's real
    ~/.config/milknado/milknado.toml never leaks into the test suite.
    Layered-config tests can still place a global file inside this dir.
    """
    xdg = tmp_path_factory.mktemp("xdg_config")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))


@pytest.fixture(autouse=True)
def _isolate_worker_identity(  # pyright: ignore[reportUnusedFunction]
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Remove ambient MILKNADO_* worker identity from the test process.

    Subprocesses inherit the cleaned environment.
    """
    for name in tuple(os.environ):
        if name.startswith("MILKNADO_"):
            monkeypatch.delenv(name, raising=False)


@pytest.fixture()
def worker_stub(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> Callable[[str], str]:
    return install_worker_stub(tmp_path_factory.mktemp("worker-stub-bin"), monkeypatch)


@pytest.fixture()
def graph(tmp_path: Path) -> Generator[MikadoGraph, None, None]:
    g = MikadoGraph(tmp_path / "test.db")
    yield g
    g.close()


@pytest.fixture()
def mock_git() -> MagicMock:
    return MagicMock(spec=GitPort)


@pytest.fixture()
def mock_crg() -> MagicMock:
    return MagicMock(spec=CrgPort)


@pytest.fixture()
def mock_ralph() -> MagicMock:
    return MagicMock(spec=LoopPort)
