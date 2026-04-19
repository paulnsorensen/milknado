from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from code_review_graph.analysis import find_bridge_nodes, find_hub_nodes
from code_review_graph.communities import get_architecture_overview, get_communities
from code_review_graph.flows import get_flows
from code_review_graph.graph import GraphStore
from code_review_graph.tools.context import get_minimal_context

_SOURCE_EXTENSIONS = frozenset({
    ".py", ".ts", ".tsx", ".js", ".jsx", ".rs", ".go",
    ".java", ".rb", ".c", ".cpp", ".h", ".hpp", ".cs",
})

_SKIP_DIRS = frozenset({
    ".git", ".code-review-graph", ".venv", ".tox", ".mypy_cache",
    ".ruff_cache", "__pycache__", "node_modules", ".next", "target",
    "dist", "build", ".eggs", "ralphs",
})


class CrgAdapter:
    def __init__(self, project_root: Path) -> None:
        self._root = project_root
        self._store: GraphStore | None = None

    def _crg_dir(self) -> Path:
        return self._root / ".code-review-graph"

    def _db_path(self) -> Path:
        return self._crg_dir() / "graph.db"

    def _get_store(self) -> GraphStore:
        if self._store is None:
            self._store = GraphStore(self._db_path())
        return self._store

    def _is_built(self) -> bool:
        return self._db_path().exists()

    def _is_stale(self) -> bool:
        db = self._db_path()
        if not db.exists():
            return False
        db_mtime = db.stat().st_mtime
        for f in self._walk_sources():
            if f.stat().st_mtime > db_mtime:
                return True
        return False

    def _walk_sources(self) -> list[Path]:
        sources: list[Path] = []
        dirs = [self._root]
        while dirs:
            current = dirs.pop()
            for child in current.iterdir():
                if child.is_dir():
                    if child.name not in _SKIP_DIRS:
                        dirs.append(child)
                elif child.suffix in _SOURCE_EXTENSIONS:
                    sources.append(child)
        return sources

    def _run_crg(self, command: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["code-review-graph", command],
            cwd=self._root,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )

    def build_graph(self, project_root: Path) -> None:
        self._root = project_root
        self._store = None
        self._run_crg("build")
        self._get_store()

    def ensure_graph(self, project_root: Path) -> None:
        self._root = project_root
        self._store = None
        if not self._is_built():
            self._run_crg("build")
        elif self._is_stale():
            self._run_crg("update")
        self._get_store()

    def get_impact_radius(self, files: list[str]) -> dict[str, Any]:
        return self._get_store().get_impact_radius(files)

    def get_architecture_overview(self) -> dict[str, Any]:
        return get_architecture_overview(self._get_store())

    def list_communities(
        self, sort_by: str = "size", min_size: int = 0,
    ) -> list[dict[str, Any]]:
        return get_communities(
            self._get_store(), sort_by=sort_by, min_size=min_size,
        )

    def list_flows(
        self, sort_by: str = "criticality", limit: int = 50,
    ) -> list[dict[str, Any]]:
        return get_flows(self._get_store(), sort_by=sort_by, limit=limit)

    def get_minimal_context(
        self,
        task: str = "",
        changed_files: list[str] | None = None,
    ) -> dict[str, Any]:
        return get_minimal_context(
            task=task,
            changed_files=changed_files,
            repo_root=str(self._root),
        )

    def get_bridge_nodes(self, top_n: int = 10) -> list[dict[str, Any]]:
        return find_bridge_nodes(self._get_store(), top_n=top_n)

    def get_hub_nodes(self, top_n: int = 10) -> list[dict[str, Any]]:
        return find_hub_nodes(self._get_store(), top_n=top_n)

    def semantic_search_nodes(
        self, query: str, top_n: int = 5,
    ) -> list[dict[str, Any]]:
        nodes = self._get_store().search_nodes(query, limit=top_n)
        return [
            {
                "name": n.name,
                "file_path": n.file_path,
                "kind": n.kind,
                "qualified_name": n.qualified_name,
            }
            for n in nodes
        ]
