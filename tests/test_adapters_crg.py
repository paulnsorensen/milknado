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


class TestGetBridgeNodes:
    @patch("milknado.adapters.crg.find_bridge_nodes")
    @patch("milknado.adapters.crg.GraphStore")
    def test_delegates_with_default_top_n(
        self,
        mock_store_cls: MagicMock,
        mock_find_bridges: MagicMock,
        adapter: CrgAdapter,
    ) -> None:
        mock_store = MagicMock()
        mock_store_cls.return_value = mock_store
        mock_find_bridges.return_value = [{"name": "auth_middleware"}]

        result = adapter.get_bridge_nodes()

        mock_find_bridges.assert_called_once_with(mock_store, top_n=10)
        assert result == [{"name": "auth_middleware"}]

    @patch("milknado.adapters.crg.find_bridge_nodes")
    @patch("milknado.adapters.crg.GraphStore")
    def test_forwards_top_n_override(
        self,
        mock_store_cls: MagicMock,
        mock_find_bridges: MagicMock,
        adapter: CrgAdapter,
    ) -> None:
        mock_store = MagicMock()
        mock_store_cls.return_value = mock_store
        mock_find_bridges.return_value = []

        adapter.get_bridge_nodes(top_n=5)

        mock_find_bridges.assert_called_once_with(mock_store, top_n=5)

    @patch("milknado.adapters.crg.find_bridge_nodes")
    @patch("milknado.adapters.crg.GraphStore")
    def test_returns_list(
        self,
        mock_store_cls: MagicMock,
        mock_find_bridges: MagicMock,
        adapter: CrgAdapter,
    ) -> None:
        mock_store = MagicMock()
        mock_store_cls.return_value = mock_store
        bridges = [{"name": "x"}, {"name": "y"}]
        mock_find_bridges.return_value = bridges

        result = adapter.get_bridge_nodes(top_n=2)

        assert result == bridges


class TestGetHubNodes:
    @patch("milknado.adapters.crg.find_hub_nodes")
    @patch("milknado.adapters.crg.GraphStore")
    def test_delegates_with_default_top_n(
        self,
        mock_store_cls: MagicMock,
        mock_find_hubs: MagicMock,
        adapter: CrgAdapter,
    ) -> None:
        mock_store = MagicMock()
        mock_store_cls.return_value = mock_store
        mock_find_hubs.return_value = [{"name": "user_service"}]

        result = adapter.get_hub_nodes()

        mock_find_hubs.assert_called_once_with(mock_store, top_n=10)
        assert result == [{"name": "user_service"}]

    @patch("milknado.adapters.crg.find_hub_nodes")
    @patch("milknado.adapters.crg.GraphStore")
    def test_forwards_top_n_override(
        self,
        mock_store_cls: MagicMock,
        mock_find_hubs: MagicMock,
        adapter: CrgAdapter,
    ) -> None:
        mock_store = MagicMock()
        mock_store_cls.return_value = mock_store
        mock_find_hubs.return_value = []

        adapter.get_hub_nodes(top_n=3)

        mock_find_hubs.assert_called_once_with(mock_store, top_n=3)

    @patch("milknado.adapters.crg.find_hub_nodes")
    @patch("milknado.adapters.crg.GraphStore")
    def test_returns_list(
        self,
        mock_store_cls: MagicMock,
        mock_find_hubs: MagicMock,
        adapter: CrgAdapter,
    ) -> None:
        mock_store = MagicMock()
        mock_store_cls.return_value = mock_store
        hubs = [{"name": "a"}, {"name": "b"}, {"name": "c"}]
        mock_find_hubs.return_value = hubs

        result = adapter.get_hub_nodes(top_n=3)

        assert result == hubs


def _make_node(name: str, file_path: str, kind: str = "function") -> MagicMock:
    n = MagicMock()
    n.name = name
    n.file_path = file_path
    n.kind = kind
    n.qualified_name = f"{file_path}::{name}"
    return n


class TestSemanticSearch:
    @patch("milknado.adapters.crg.GraphStore")
    def test_minimal_returns_only_file_paths(
        self, mock_store_cls: MagicMock, adapter: CrgAdapter
    ) -> None:
        mock_store = MagicMock()
        mock_store_cls.return_value = mock_store
        mock_store.search_nodes.return_value = [
            _make_node("plan", "src/planning/planner.py"),
            _make_node("build_context", "src/planning/context.py"),
        ]

        result = adapter.semantic_search("planning context", top_n=5)

        assert result == [
            {"file_path": "src/planning/planner.py"},
            {"file_path": "src/planning/context.py"},
        ]
        mock_store.search_nodes.assert_called_once_with("planning context", limit=5)

    @patch("milknado.adapters.crg.GraphStore")
    def test_minimal_deduplicates_file_paths(
        self, mock_store_cls: MagicMock, adapter: CrgAdapter
    ) -> None:
        mock_store = MagicMock()
        mock_store_cls.return_value = mock_store
        mock_store.search_nodes.return_value = [
            _make_node("plan", "src/planning/planner.py"),
            _make_node("replan", "src/planning/planner.py"),
            _make_node("build_context", "src/planning/context.py"),
        ]

        result = adapter.semantic_search("plan context", detail_level="minimal")

        assert result == [
            {"file_path": "src/planning/planner.py"},
            {"file_path": "src/planning/context.py"},
        ]

    @patch("milknado.adapters.crg.GraphStore")
    def test_full_returns_all_node_fields(
        self, mock_store_cls: MagicMock, adapter: CrgAdapter
    ) -> None:
        mock_store = MagicMock()
        mock_store_cls.return_value = mock_store
        mock_store.search_nodes.return_value = [
            _make_node("plan", "src/planning/planner.py", kind="function"),
        ]

        result = adapter.semantic_search("plan nodes", detail_level="full")

        assert result == [
            {
                "name": "plan",
                "file_path": "src/planning/planner.py",
                "kind": "function",
                "qualified_name": "src/planning/planner.py::plan",
            }
        ]

    @patch("milknado.adapters.crg.GraphStore")
    def test_empty_results(
        self, mock_store_cls: MagicMock, adapter: CrgAdapter
    ) -> None:
        mock_store = MagicMock()
        mock_store_cls.return_value = mock_store
        mock_store.search_nodes.return_value = []

        result = adapter.semantic_search("no match here")

        assert result == []

    @patch("milknado.adapters.crg.GraphStore")
    def test_default_detail_level_is_minimal(
        self, mock_store_cls: MagicMock, adapter: CrgAdapter
    ) -> None:
        mock_store = MagicMock()
        mock_store_cls.return_value = mock_store
        mock_store.search_nodes.return_value = [
            _make_node("foo", "src/foo.py"),
        ]

        result = adapter.semantic_search("foo bar")

        assert list(result[0].keys()) == ["file_path"]

    @patch("milknado.adapters.crg.GraphStore")
    def test_default_top_n_is_five(
        self, mock_store_cls: MagicMock, adapter: CrgAdapter
    ) -> None:
        mock_store = MagicMock()
        mock_store_cls.return_value = mock_store
        mock_store.search_nodes.return_value = []

        adapter.semantic_search("foo bar")

        mock_store.search_nodes.assert_called_once_with("foo bar", limit=5)

    def test_phrase_cap_rejects_single_word(self, adapter: CrgAdapter) -> None:
        with pytest.raises(ValueError, match="2-4 words"):
            adapter.semantic_search("plan")

    @patch("milknado.adapters.crg.GraphStore")
    def test_phrase_cap_truncates_long_query(
        self, mock_store_cls: MagicMock, adapter: CrgAdapter
    ) -> None:
        mock_store = MagicMock()
        mock_store_cls.return_value = mock_store
        mock_store.search_nodes.return_value = []

        adapter.semantic_search("one two three four five six")

        mock_store.search_nodes.assert_called_once_with("one two three four", limit=5)
