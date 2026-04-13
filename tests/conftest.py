from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from milknado.domains.common.protocols import CrgPort, GitPort, RalphPort
from milknado.domains.graph import MikadoGraph


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
    return MagicMock(spec=RalphPort)
