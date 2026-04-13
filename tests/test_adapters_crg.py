from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from milknado.adapters.crg import CrgAdapter


@pytest.fixture()
def adapter(tmp_path: Path) -> CrgAdapter:
    return CrgAdapter(tmp_path)


class TestEnsureGraph:
    @patch("milknado.adapters.crg.GraphStore")
    def test_creates_store_for_project(
        self, mock_store_cls: MagicMock, tmp_path: Path
    ) -> None:
        adapter = CrgAdapter(tmp_path)
        new_root = tmp_path / "other"
        adapter.ensure_graph(new_root)
        expected_db = new_root / ".code-review-graph" / "graph.db"
        mock_store_cls.assert_called_once_with(expected_db)

    @patch("milknado.adapters.crg.GraphStore")
    def test_resets_cached_store(
        self, mock_store_cls: MagicMock, adapter: CrgAdapter
    ) -> None:
        adapter._get_store()
        adapter.ensure_graph(adapter._root)
        assert mock_store_cls.call_count == 2


class TestGetImpactRadius:
    @patch("milknado.adapters.crg.GraphStore")
    def test_delegates_to_store(
        self, mock_store_cls: MagicMock, adapter: CrgAdapter
    ) -> None:
        mock_store = MagicMock()
        mock_store.get_impact_radius.return_value = {"nodes": []}
        mock_store_cls.return_value = mock_store

        result = adapter.get_impact_radius(["src/foo.py"])
        mock_store.get_impact_radius.assert_called_once_with(["src/foo.py"])
        assert result == {"nodes": []}


class TestGetArchitectureOverview:
    @patch("milknado.adapters.crg.get_architecture_overview")
    @patch("milknado.adapters.crg.GraphStore")
    def test_delegates_to_communities_module(
        self,
        mock_store_cls: MagicMock,
        mock_get_arch: MagicMock,
        adapter: CrgAdapter,
    ) -> None:
        mock_store = MagicMock()
        mock_store_cls.return_value = mock_store
        mock_get_arch.return_value = {"communities": []}

        result = adapter.get_architecture_overview()
        mock_get_arch.assert_called_once_with(mock_store)
        assert result == {"communities": []}
