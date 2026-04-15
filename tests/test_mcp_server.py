from __future__ import annotations

import os
from pathlib import Path

from milknado.mcp_server import _open_graph, _project_root


class TestMcpServer:
    def test_open_graph_creates_default_db_dir(self, tmp_path: Path) -> None:
        graph, _cfg = _open_graph(tmp_path)
        try:
            assert graph.get_all_nodes() == []
        finally:
            graph.close()
        assert (tmp_path / ".milknado" / "milknado.db").exists()

    def test_open_graph_and_add_node(self, tmp_path: Path) -> None:
        graph, _cfg = _open_graph(tmp_path)
        try:
            node = graph.add_node("first task")
            nodes = graph.get_all_nodes()
        finally:
            graph.close()
        assert node.id == 1
        assert len(nodes) == 1
        assert nodes[0].description == "first task"

    def test_project_root_precedence(self, tmp_path: Path, monkeypatch) -> None:
        explicit = tmp_path / "explicit"
        from_env = tmp_path / "from-env"
        monkeypatch.setenv("MILKNADO_PROJECT_ROOT", str(from_env))

        assert _project_root(str(explicit)) == explicit.resolve()
        assert _project_root(None) == from_env.resolve()

    def test_project_root_falls_back_to_cwd(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.delenv("MILKNADO_PROJECT_ROOT", raising=False)
        old = Path.cwd()
        os.chdir(tmp_path)
        try:
            assert _project_root(None) == tmp_path.resolve()
        finally:
            os.chdir(old)
