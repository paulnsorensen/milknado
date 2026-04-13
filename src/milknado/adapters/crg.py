from __future__ import annotations

from pathlib import Path
from typing import Any

from code_review_graph.communities import get_architecture_overview
from code_review_graph.graph import GraphStore


class CrgAdapter:
    def __init__(self, project_root: Path) -> None:
        self._root = project_root
        self._store: GraphStore | None = None

    def _db_path(self) -> Path:
        return self._root / ".code-review-graph" / "graph.db"

    def _get_store(self) -> GraphStore:
        if self._store is None:
            self._store = GraphStore(self._db_path())
        return self._store

    def ensure_graph(self, project_root: Path) -> None:
        self._root = project_root
        self._store = None
        self._get_store()

    def get_impact_radius(self, files: list[str]) -> dict[str, Any]:
        return self._get_store().get_impact_radius(files)

    def get_architecture_overview(self) -> dict[str, Any]:
        return get_architecture_overview(self._get_store())
