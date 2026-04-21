from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from milknado.domains.common.types import RebaseResult
from milknado.domains.execution import ExecutionConfig, Executor, RunLoop
from milknado.domains.graph import MikadoGraph


@dataclass
class _RunState:
    run_id: str = "run-1"


@dataclass
class _Run:
    state: _RunState = field(default_factory=_RunState)


class _FakeGit:
    def create_worktree(self, path: Path, branch: str) -> Path:  # noqa: ARG002
        path.mkdir(parents=True, exist_ok=True)
        return path

    def remove_worktree(self, path: Path) -> None:  # noqa: ARG002
        pass

    def rebase(self, worktree: Path, onto: str) -> RebaseResult:  # noqa: ARG002
        return RebaseResult(success=True)

    def current_branch(self) -> str:
        return "main"

    def commit_all(self, worktree: Path, message: str) -> None:  # noqa: ARG002
        pass

    def squash_and_commit(self, worktree: Path, onto: str, msg: str) -> None:  # noqa: ARG002
        pass


class _FakeCrg:
    def ensure_graph(self, project_root: Path) -> None:  # noqa: ARG002
        pass

    def get_impact_radius(self, files: list[str]) -> dict[str, Any]:
        return {}

    def get_architecture_overview(self) -> dict[str, Any]:
        return {}

    def list_communities(
        self,
        sort_by: str = "size",
        min_size: int = 0,
    ) -> list[dict[str, Any]]:
        return []

    def list_flows(
        self,
        sort_by: str = "criticality",
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        return []

    def get_minimal_context(
        self,
        task: str = "",
        changed_files: list[str] | None = None,
    ) -> dict[str, Any]:
        return {}

    def get_bridge_nodes(self, top_n: int = 10) -> list[dict[str, Any]]:
        return []

    def get_hub_nodes(self, top_n: int = 10) -> list[dict[str, Any]]:
        return []

    def semantic_search_nodes(
        self,
        query: str,
        top_n: int = 5,
    ) -> list[dict[str, Any]]:
        return []

    def semantic_search(
        self,
        query: str,
        top_n: int = 5,
        detail_level: str = "minimal",
    ) -> list[dict[str, Any]]:
        return []


class _FakeRalph:
    def __init__(self) -> None:
        self._counter = 0
        self._pending: list[tuple[str, bool]] = []

    def create_run(
        self,
        agent: str,
        ralph_dir: Path,
        ralph_file: Path,
        commands: list[str],
        quality_gates: list[str],
        project_root: Path | None = None,
    ) -> _Run:
        self._counter += 1
        run_id = f"run-{self._counter}"
        self._pending.append((run_id, True))
        return _Run(state=_RunState(run_id=run_id))

    def start_run(self, run_id: str) -> None:
        pass

    def stop_run(self, run_id: str) -> None:
        pass

    def list_runs(self) -> list[Any]:
        return []

    def get_run(self, run_id: str) -> Any | None:
        return None

    def get_run_stdout(self, run_id: str) -> list[str]:
        return []

    def wait_for_next_completion(
        self,
        active_run_ids: set[str],
        timeout: float | None = None,
    ) -> tuple[str, bool]:
        for i, (run_id, success) in enumerate(self._pending):
            if run_id in active_run_ids:
                self._pending.pop(i)
                return run_id, success
        raise RuntimeError("No pending completions for active runs")

    def poll_progress_events(self) -> list[Any]:
        return []

    def verify_spec(self, spec_text: str, graph_state: str) -> Any:
        from milknado.domains.common.protocols import VerifySpecResult

        return VerifySpecResult(outcome="done")

    def generate_ralph_md(
        self,
        node: Any,
        context: str,
        quality_gates: list[str],
        output_path: Path,
    ) -> Path:
        return output_path


@pytest.fixture()
def transcript_config(tmp_path: Path) -> ExecutionConfig:
    return ExecutionConfig(
        execution_agent="claude",
        quality_gates=("uv run pytest",),
        worktree_pattern="milknado-{node_id}-{slug}",
        project_root=tmp_path,
    )


class TestTranscriptContract:
    """US-201: run log is non-empty and terminated with FINAL_TELEMETRY."""

    def test_log_file_created_under_milknado_dir(
        self,
        graph: MikadoGraph,
        transcript_config: ExecutionConfig,
        tmp_path: Path,
    ) -> None:
        ralph = _FakeRalph()
        executor = Executor(
            graph=graph,
            git=_FakeGit(),
            ralph=ralph,
            crg=_FakeCrg(),
        )
        loop = RunLoop(executor=executor, graph=graph, ralph=ralph)
        graph.add_node("single node")

        loop.run(transcript_config, "main")

        log_files = sorted((tmp_path / ".milknado").glob("run-*.log"))
        assert log_files, "expected at least one run-*.log in .milknado/"

    def test_log_file_at_least_200_bytes(
        self,
        graph: MikadoGraph,
        transcript_config: ExecutionConfig,
        tmp_path: Path,
    ) -> None:
        ralph = _FakeRalph()
        executor = Executor(
            graph=graph,
            git=_FakeGit(),
            ralph=ralph,
            crg=_FakeCrg(),
        )
        loop = RunLoop(executor=executor, graph=graph, ralph=ralph)
        graph.add_node("single node")

        loop.run(transcript_config, "main")

        log_path = sorted((tmp_path / ".milknado").glob("run-*.log"))[-1]
        assert log_path.stat().st_size >= 200

    def test_log_contains_node_dispatched_event(
        self,
        graph: MikadoGraph,
        transcript_config: ExecutionConfig,
        tmp_path: Path,
    ) -> None:
        ralph = _FakeRalph()
        executor = Executor(
            graph=graph,
            git=_FakeGit(),
            ralph=ralph,
            crg=_FakeCrg(),
        )
        loop = RunLoop(executor=executor, graph=graph, ralph=ralph)
        root = graph.add_node("root")
        graph.add_node("single node", parent_id=root.id)

        loop.run(transcript_config, "main")

        log_path = sorted((tmp_path / ".milknado").glob("run-*.log"))[-1]
        content = log_path.read_text(encoding="utf-8")
        assert "node_dispatched" in content

    def test_log_contains_node_completed_event(
        self,
        graph: MikadoGraph,
        transcript_config: ExecutionConfig,
        tmp_path: Path,
    ) -> None:
        ralph = _FakeRalph()
        executor = Executor(
            graph=graph,
            git=_FakeGit(),
            ralph=ralph,
            crg=_FakeCrg(),
        )
        loop = RunLoop(executor=executor, graph=graph, ralph=ralph)
        root = graph.add_node("root")
        graph.add_node("single node", parent_id=root.id)

        loop.run(transcript_config, "main")

        log_path = sorted((tmp_path / ".milknado").glob("run-*.log"))[-1]
        content = log_path.read_text(encoding="utf-8")
        assert "node_completed" in content

    def test_last_log_line_is_final_telemetry(
        self,
        graph: MikadoGraph,
        transcript_config: ExecutionConfig,
        tmp_path: Path,
    ) -> None:
        ralph = _FakeRalph()
        executor = Executor(
            graph=graph,
            git=_FakeGit(),
            ralph=ralph,
            crg=_FakeCrg(),
        )
        loop = RunLoop(executor=executor, graph=graph, ralph=ralph)
        graph.add_node("single node")

        loop.run(transcript_config, "main")

        log_path = sorted((tmp_path / ".milknado").glob("run-*.log"))[-1]
        content = log_path.read_text(encoding="utf-8")
        lines = [ln for ln in content.splitlines() if ln.strip()]
        assert lines, "log file has no non-empty lines"
        assert "FINAL_TELEMETRY {" in lines[-1]


class TestTranscriptFormat:
    """US-202: filename and per-line format contract."""

    _FILENAME_RE = re.compile(r"^run-\d{8}T\d{6}Z\.log$")
    _LINE_RE = re.compile(
        r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} "
        r"(INFO|WARNING|ERROR) milknado(\.[^:]+)?: .+$"
    )

    def _smoke_run(self, graph: MikadoGraph, config: ExecutionConfig) -> Path:
        ralph = _FakeRalph()
        executor = Executor(graph=graph, git=_FakeGit(), ralph=ralph, crg=_FakeCrg())
        loop = RunLoop(executor=executor, graph=graph, ralph=ralph)
        for i in range(3):
            graph.add_node(f"node {i}")
        loop.run(config, "main")
        log_files = sorted((config.project_root / ".milknado").glob("run-*.log"))
        assert log_files, "smoke run produced no log file"
        return log_files[-1]

    def test_filename_matches_utc_iso8601_pattern(
        self,
        graph: MikadoGraph,
        transcript_config: ExecutionConfig,
    ) -> None:
        log_path = self._smoke_run(graph, transcript_config)
        assert self._FILENAME_RE.match(log_path.name), (
            f"filename {log_path.name!r} does not match ^run-\\d{{8}}T\\d{{6}}Z\\.log$"
        )

    def test_log_lines_match_format(
        self,
        graph: MikadoGraph,
        transcript_config: ExecutionConfig,
    ) -> None:
        log_path = self._smoke_run(graph, transcript_config)
        content = log_path.read_text(encoding="utf-8")
        lines = [ln for ln in content.splitlines() if ln.strip()]
        assert len(lines) >= 5, f"expected >= 5 log lines, got {len(lines)}"
        for ln in lines[:5]:
            assert self._LINE_RE.match(ln), f"log line does not match format contract: {ln!r}"
