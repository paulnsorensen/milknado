from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from milknado.adapters.crg import CrgAdapter


@pytest.fixture()
def adapter(tmp_path: Path) -> CrgAdapter:
    return CrgAdapter(tmp_path)


def _make_crg_db(root: Path) -> Path:
    crg_dir = root / ".code-review-graph"
    crg_dir.mkdir(parents=True, exist_ok=True)
    db = crg_dir / "graph.db"
    db.touch()
    return db


class TestIsBuilt:
    def test_false_when_no_crg_dir(self, adapter: CrgAdapter) -> None:
        assert adapter._is_built() is False

    def test_true_when_db_exists(self, tmp_path: Path) -> None:
        _make_crg_db(tmp_path)
        adapter = CrgAdapter(tmp_path)
        assert adapter._is_built() is True


class TestIsStale:
    def test_false_when_no_db(self, adapter: CrgAdapter) -> None:
        assert adapter._is_stale() is False

    def test_false_when_no_sources(self, tmp_path: Path) -> None:
        _make_crg_db(tmp_path)
        adapter = CrgAdapter(tmp_path)
        assert adapter._is_stale() is False

    def test_false_when_sources_older_than_db(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        src.mkdir()
        (src / "main.py").write_text("x = 1")
        time.sleep(0.05)
        _make_crg_db(tmp_path)
        adapter = CrgAdapter(tmp_path)
        assert adapter._is_stale() is False

    def test_true_when_source_newer_than_db(self, tmp_path: Path) -> None:
        db = _make_crg_db(tmp_path)
        db_mtime = db.stat().st_mtime
        src = tmp_path / "src"
        src.mkdir()
        py_file = src / "main.py"
        py_file.write_text("x = 1")
        import os
        os.utime(py_file, (db_mtime + 10, db_mtime + 10))
        adapter = CrgAdapter(tmp_path)
        assert adapter._is_stale() is True

    def test_detects_stale_outside_src(self, tmp_path: Path) -> None:
        db = _make_crg_db(tmp_path)
        db_mtime = db.stat().st_mtime
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        py_file = tests_dir / "test_foo.py"
        py_file.write_text("assert True")
        import os
        os.utime(py_file, (db_mtime + 10, db_mtime + 10))
        adapter = CrgAdapter(tmp_path)
        assert adapter._is_stale() is True

    def test_detects_stale_at_project_root(self, tmp_path: Path) -> None:
        db = _make_crg_db(tmp_path)
        db_mtime = db.stat().st_mtime
        py_file = tmp_path / "setup.py"
        py_file.write_text("setup()")
        import os
        os.utime(py_file, (db_mtime + 10, db_mtime + 10))
        adapter = CrgAdapter(tmp_path)
        assert adapter._is_stale() is True

    def test_ignores_non_source_files(self, tmp_path: Path) -> None:
        db = _make_crg_db(tmp_path)
        db_mtime = db.stat().st_mtime
        src = tmp_path / "src"
        src.mkdir()
        txt = src / "notes.txt"
        txt.write_text("not source")
        import os
        os.utime(txt, (db_mtime + 10, db_mtime + 10))
        adapter = CrgAdapter(tmp_path)
        assert adapter._is_stale() is False

    def test_skips_excluded_dirs(self, tmp_path: Path) -> None:
        db = _make_crg_db(tmp_path)
        db_mtime = db.stat().st_mtime
        venv = tmp_path / ".venv" / "lib"
        venv.mkdir(parents=True)
        py_file = venv / "site.py"
        py_file.write_text("x = 1")
        import os
        os.utime(py_file, (db_mtime + 10, db_mtime + 10))
        adapter = CrgAdapter(tmp_path)
        assert adapter._is_stale() is False


class TestEnsureGraph:
    @patch("milknado.adapters.crg.GraphStore")
    @patch("milknado.adapters.crg.subprocess.run")
    def test_builds_when_not_built(
        self, mock_run: MagicMock, mock_store_cls: MagicMock, tmp_path: Path
    ) -> None:
        adapter = CrgAdapter(tmp_path)
        adapter.ensure_graph(tmp_path)
        mock_run.assert_called_once()
        assert mock_run.call_args[0][0] == ["code-review-graph", "build"]

    @patch("milknado.adapters.crg.GraphStore")
    @patch("milknado.adapters.crg.subprocess.run")
    def test_updates_when_stale(
        self, mock_run: MagicMock, mock_store_cls: MagicMock, tmp_path: Path
    ) -> None:
        db = _make_crg_db(tmp_path)
        db_mtime = db.stat().st_mtime
        src = tmp_path / "src"
        src.mkdir()
        py = src / "app.py"
        py.write_text("x = 1")
        import os
        os.utime(py, (db_mtime + 10, db_mtime + 10))

        adapter = CrgAdapter(tmp_path)
        adapter.ensure_graph(tmp_path)
        mock_run.assert_called_once()
        assert mock_run.call_args[0][0] == ["code-review-graph", "update"]

    @patch("milknado.adapters.crg.GraphStore")
    @patch("milknado.adapters.crg.subprocess.run")
    def test_skips_when_built_and_fresh(
        self, mock_run: MagicMock, mock_store_cls: MagicMock, tmp_path: Path
    ) -> None:
        _make_crg_db(tmp_path)
        adapter = CrgAdapter(tmp_path)
        adapter.ensure_graph(tmp_path)
        mock_run.assert_not_called()

    @patch("milknado.adapters.crg.GraphStore")
    @patch("milknado.adapters.crg.subprocess.run")
    def test_resets_cached_store(
        self, mock_run: MagicMock, mock_store_cls: MagicMock, tmp_path: Path
    ) -> None:
        _make_crg_db(tmp_path)
        adapter = CrgAdapter(tmp_path)
        adapter._get_store()
        adapter.ensure_graph(tmp_path)
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


class TestListCommunities:
    @patch("milknado.adapters.crg.get_communities")
    @patch("milknado.adapters.crg.GraphStore")
    def test_delegates_with_defaults(
        self,
        mock_store_cls: MagicMock,
        mock_get_communities: MagicMock,
        adapter: CrgAdapter,
    ) -> None:
        mock_store = MagicMock()
        mock_store_cls.return_value = mock_store
        mock_get_communities.return_value = [{"id": "c1"}]

        result = adapter.list_communities()

        mock_get_communities.assert_called_once_with(
            mock_store, sort_by="size", min_size=0,
        )
        assert result == [{"id": "c1"}]

    @patch("milknado.adapters.crg.get_communities")
    @patch("milknado.adapters.crg.GraphStore")
    def test_forwards_overrides(
        self,
        mock_store_cls: MagicMock,
        mock_get_communities: MagicMock,
        adapter: CrgAdapter,
    ) -> None:
        mock_store = MagicMock()
        mock_store_cls.return_value = mock_store
        mock_get_communities.return_value = []

        adapter.list_communities(sort_by="name", min_size=5)

        mock_get_communities.assert_called_once_with(
            mock_store, sort_by="name", min_size=5,
        )


class TestListFlows:
    @patch("milknado.adapters.crg.get_flows")
    @patch("milknado.adapters.crg.GraphStore")
    def test_delegates_with_defaults(
        self,
        mock_store_cls: MagicMock,
        mock_get_flows: MagicMock,
        adapter: CrgAdapter,
    ) -> None:
        mock_store = MagicMock()
        mock_store_cls.return_value = mock_store
        mock_get_flows.return_value = [{"id": "f1"}]

        result = adapter.list_flows()

        mock_get_flows.assert_called_once_with(
            mock_store, sort_by="criticality", limit=50,
        )
        assert result == [{"id": "f1"}]

    @patch("milknado.adapters.crg.get_flows")
    @patch("milknado.adapters.crg.GraphStore")
    def test_forwards_overrides(
        self,
        mock_store_cls: MagicMock,
        mock_get_flows: MagicMock,
        adapter: CrgAdapter,
    ) -> None:
        mock_store = MagicMock()
        mock_store_cls.return_value = mock_store
        mock_get_flows.return_value = []

        adapter.list_flows(sort_by="name", limit=10)

        mock_get_flows.assert_called_once_with(
            mock_store, sort_by="name", limit=10,
        )


class TestGetMinimalContext:
    @patch("milknado.adapters.crg.get_minimal_context")
    def test_passes_repo_root_and_args(
        self,
        mock_get_context: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_get_context.return_value = {"files": []}
        adapter = CrgAdapter(tmp_path)

        result = adapter.get_minimal_context(
            task="refactor auth",
            changed_files=["src/auth.py"],
        )

        mock_get_context.assert_called_once_with(
            task="refactor auth",
            changed_files=["src/auth.py"],
            repo_root=str(tmp_path),
        )
        assert result == {"files": []}

    @patch("milknado.adapters.crg.get_minimal_context")
    def test_defaults(
        self,
        mock_get_context: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_get_context.return_value = {}
        adapter = CrgAdapter(tmp_path)

        adapter.get_minimal_context()

        mock_get_context.assert_called_once_with(
            task="",
            changed_files=None,
            repo_root=str(tmp_path),
        )
