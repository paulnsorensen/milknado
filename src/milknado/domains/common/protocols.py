from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from milknado.domains.common.types import (
    DegradationMarker,
    MikadoNode,
    RebaseResult,
    TilthMap,
)


@dataclass(frozen=True)
class ProgressEvent:
    run_id: str
    work: int
    total: int
    message: str = ""


@dataclass(frozen=True)
class VerifySpecResult:
    outcome: Literal["done", "gaps"]
    goal_delta: str | None = None


@dataclass(frozen=True)
class SymbolLocation:
    path: Path
    line_start: int
    line_end: int


class GitPort(Protocol):
    def create_worktree(self, path: Path, branch: str) -> Path: ...
    def branch_exists(self, branch: str) -> bool: ...
    def remove_worktree(self, path: Path, target: str = "HEAD") -> None: ...
    def force_remove_worktree(self, path: Path) -> None: ...
    def rebase(self, worktree: Path, onto: str) -> RebaseResult: ...
    def current_branch(self) -> str: ...
    def resolve_ref(self, ref: str) -> str: ...
    def diff_for_review(self, worktree: Path, base_oid: str) -> str: ...
    def compare_and_swap_ref(self, ref: str, expected_oid: str, new_oid: str) -> None: ...
    def commit_all(self, worktree: Path, message: str) -> None: ...
    def squash_and_commit(self, worktree: Path, onto: str, msg: str) -> bool: ...
    def fast_forward(self, branch: str) -> None: ...


class TilthPort(Protocol):
    def structural_map(
        self,
        scope: Path,
        budget_tokens: int,
    ) -> TilthMap | DegradationMarker: ...

    def search_symbol(
        self,
        keyword: str,
        glob: str | None = None,
    ) -> list[SymbolLocation]: ...

    def read_section(self, path: Path, line_start: int, line_end: int) -> str: ...


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
    def get_bridge_nodes(self, top_n: int = 10) -> list[dict[str, Any]]: ...
    def get_hub_nodes(self, top_n: int = 10) -> list[dict[str, Any]]: ...


class CommandResult(Protocol):
    returncode: int
    stdout: str
    stderr: str


class ToolchainPort(Protocol):
    def find_executable(self, command: str) -> str | None: ...
    def run(self, argv: list[str], *, check: bool = False) -> CommandResult: ...
    def system(self) -> str: ...
    def machine(self) -> str: ...
    def environment(self, name: str) -> str | None: ...
    def home(self) -> Path: ...


TerminalRunOutcome = Literal["completed", "stopped", "failed"]


class LoopPort(Protocol):
    def create_run(
        self,
        agent: str,
        ralph_dir: Path,
        ralph_file: Path,
        commands: list[str],
        quality_gates: list[str],
        project_root: Path | None = None,
        commit_footer: str | None = None,
        runtime_policy: Any | None = None,
        run_id: str | None = None,
    ) -> Any: ...
    def start_run(self, run_id: str) -> None: ...
    def queue_guidance(self, run_id: str, text: str) -> bool: ...
    def request_stop_run(self, run_id: str) -> None: ...
    def stop_run(self, run_id: str, timeout: float | None = None) -> bool: ...
    def force_stop_run(self, run_id: str, timeout: float | None = None) -> bool: ...
    def list_runs(self) -> list[Any]: ...
    def get_run(self, run_id: str) -> Any | None: ...
    def get_run_stdout(self, run_id: str) -> list[str]: ...
    def get_run_failure_detail(self, run_id: str) -> str | None: ...
    def get_run_output_tail(self, run_id: str, max_lines: int) -> list[str]: ...
    def get_run_guidance(self, run_id: str) -> tuple[str, ...]: ...
    def wait_for_next_completion(
        self,
        active_run_ids: set[str],
        timeout: float | None = None,
    ) -> tuple[str, TerminalRunOutcome | ProgressEvent]: ...
    def run_node_review(
        self,
        agent: str,
        prompt: str,
        worktree: Path,
        project_root: Path,
    ) -> Any: ...
    def verify_spec(
        self,
        spec_text: str,
        graph_state: str,
    ) -> VerifySpecResult: ...
    def generate_ralph_md(
        self,
        node: MikadoNode,
        context: str,
        quality_gates: list[str],
        output_path: Path,
        prior_findings: str = "",
        findings_round: int | None = None,
    ) -> Path: ...
