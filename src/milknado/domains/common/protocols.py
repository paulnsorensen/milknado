from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from milknado.domains.common.types import (
    DegradationMarker,
    MikadoNode,
    RebaseResult,
    TilthMap,
)


class GitPort(Protocol):
    def create_worktree(self, path: Path, branch: str) -> Path: ...
    def remove_worktree(self, path: Path) -> None: ...
    def rebase(self, worktree: Path, onto: str) -> RebaseResult: ...
    def current_branch(self) -> str: ...
    def commit_all(self, worktree: Path, message: str) -> None: ...


class TilthPort(Protocol):
    def structural_map(
        self,
        scope: Path,
        budget_tokens: int,
    ) -> TilthMap | DegradationMarker: ...


class CrgPort(Protocol):
    def ensure_graph(self, project_root: Path) -> None: ...
    def get_impact_radius(self, files: list[str]) -> dict[str, Any]: ...
    def get_architecture_overview(self) -> dict[str, Any]: ...
    def list_communities(
        self,
        sort_by: str = "size",
        min_size: int = 0,
    ) -> list[dict[str, Any]]: ...
    def list_flows(
        self,
        sort_by: str = "criticality",
        limit: int = 50,
    ) -> list[dict[str, Any]]: ...
    def get_minimal_context(
        self,
        task: str = "",
        changed_files: list[str] | None = None,
    ) -> dict[str, Any]: ...
    def get_bridge_nodes(self, top_n: int = 10) -> list[dict[str, Any]]: ...
    def get_hub_nodes(self, top_n: int = 10) -> list[dict[str, Any]]: ...


class RalphPort(Protocol):
    def create_run(
        self,
        agent: str,
        ralph_dir: Path,
        ralph_file: Path,
        commands: list[str],
        quality_gates: list[str],
        project_root: Path | None = None,
    ) -> Any: ...
    def start_run(self, run_id: str) -> None: ...
    def stop_run(self, run_id: str) -> None: ...
    def list_runs(self) -> list[Any]: ...
    def get_run(self, run_id: str) -> Any | None: ...
    def wait_for_next_completion(
        self,
        active_run_ids: set[str],
    ) -> tuple[str, bool]: ...
    def generate_ralph_md(
        self,
        node: MikadoNode,
        context: str,
        quality_gates: list[str],
        output_path: Path,
    ) -> Path: ...
