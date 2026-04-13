from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from code_review_graph.communities import get_architecture_overview
from code_review_graph.graph import GraphStore

_SOURCE_EXTENSIONS = frozenset({
    ".py", ".ts", ".tsx", ".js", ".jsx", ".rs", ".go",
    ".java", ".rb", ".c", ".cpp", ".h", ".hpp", ".cs",
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
        src = self._root / "src"
        if not src.exists():
            return False
        for f in src.rglob("*"):
            if f.suffix in _SOURCE_EXTENSIONS and f.stat().st_mtime > db_mtime:
                return True
        return False

    def _run_crg(self, command: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["code-review-graph", command],
            cwd=self._root,
            capture_output=True,
            text=True,
            check=False,
        )

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
