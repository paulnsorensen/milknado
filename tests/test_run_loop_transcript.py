from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from milknado.domains.common.types import MikadoNode, RebaseResult
from milknado.domains.execution import ExecutionConfig, Executor, RunLoop
from milknado.domains.graph import MikadoGraph

# ── Fakes ─────────────────────────────────────────────────────────────────────


@dataclass
class _FakeRunState:
    run_id: str


@dataclass
class _FakeRun:
    state: _FakeRunState


class _FakeGit:
    def __init__(self) -> None:
        self.rebase_result: RebaseResult = RebaseResult(success=True)

    def create_worktree(self, path: Path, branch: str) -> Path:
        path.mkdir(parents=True, exist_ok=True)
        return path

    def remove_worktree(self, path: Path) -> None:
        pass

    def rebase(self, worktree: Path, onto: str) -> RebaseResult:
        return self.rebase_result

    def current_branch(self) -> str:
        return "main"

    def commit_all(self, worktree: Path, message: str) -> None:
        pass

    def squash_and_commit(self, worktree: Path, onto: str, msg: str) -> None:
        pass


class _FakeCrg:
    def ensure_graph(self, project_root: Path) -> None:
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

    def semantic_search_nodes(self, query: str, top_n: int = 5) -> list[dict[str, Any]]:
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
        self._run_counter = 0
        self._success: dict[str, bool] = {}
        self._pending_completions: list[tuple[str, bool]] = []
        self.raise_interrupt: bool = False

    def create_run(
        self,
        agent: str,
        ralph_dir: Path,
        ralph_file: Path,
        commands: list[str],
        quality_gates: list[str],
        project_root: Path | None = None,
    ) -> _FakeRun:
        self._run_counter += 1
        run_id = f"run-{self._run_counter}"
        success = self._success.get(run_id, True)
        self._pending_completions.append((run_id, success))
        return _FakeRun(state=_FakeRunState(run_id=run_id))

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
        if self.raise_interrupt:
            raise KeyboardInterrupt
        for i, (run_id, success) in enumerate(self._pending_completions):
            if run_id in active_run_ids:
                self._pending_completions.pop(i)
                return run_id, success
        raise RuntimeError("No pending completions for active runs")

    def poll_progress_events(self) -> list[Any]:
        return []

    def verify_spec(self, spec_text: str, graph_state: str) -> Any:
        from milknado.domains.common.protocols import VerifySpecResult

        return VerifySpecResult(outcome="done")

    def generate_ralph_md(
        self,
        node: MikadoNode,
        context: str,
        quality_gates: list[str],
        output_path: Path,
    ) -> Path:
        return output_path


# ── Helpers ───────────────────────────────────────────────────────────────────


def _find_log_file(project_root: Path) -> Path | None:
    log_dir = project_root / ".milknado"
    if not log_dir.exists():
        return None
    logs = sorted(log_dir.glob("run-*.log"))
    return logs[-1] if logs else None


def _make_run_loop(
    graph: MikadoGraph,
    ralph: _FakeRalph,
) -> RunLoop:
    executor = Executor(graph=graph, git=_FakeGit(), ralph=ralph, crg=_FakeCrg())
    return RunLoop(executor=executor, graph=graph, ralph=ralph)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def exec_config(tmp_path: Path) -> ExecutionConfig:
    return ExecutionConfig(
        execution_agent="claude",
        quality_gates=("uv run pytest",),
        worktree_pattern="milknado-{node_id}-{slug}",
        project_root=tmp_path,
    )


@pytest.fixture()
def graph(tmp_path: Path) -> Generator[MikadoGraph, None, None]:
    g = MikadoGraph(tmp_path / "test.db")
    yield g
    g.close()


@pytest.fixture()
def fake_ralph() -> _FakeRalph:
    return _FakeRalph()


@pytest.fixture()
def run_loop(graph: MikadoGraph, fake_ralph: _FakeRalph) -> RunLoop:
    return _make_run_loop(graph, fake_ralph)


# ── Tests: file created on run start ─────────────────────────────────────────


class TestLogFileCreated:
    def test_log_file_exists_after_run(
        self,
        run_loop: RunLoop,
        graph: MikadoGraph,
        exec_config: ExecutionConfig,
        tmp_path: Path,
    ) -> None:
        graph.add_node("only-node")
        run_loop.run(exec_config, "main")

        log_file = _find_log_file(tmp_path)
        assert log_file is not None
        assert log_file.exists()

    def test_log_file_in_milknado_dir(
        self,
        run_loop: RunLoop,
        graph: MikadoGraph,
        exec_config: ExecutionConfig,
        tmp_path: Path,
    ) -> None:
        graph.add_node("only-node")
        run_loop.run(exec_config, "main")

        log_file = _find_log_file(tmp_path)
        assert log_file is not None
        assert log_file.parent == tmp_path / ".milknado"
        assert log_file.name.startswith("run-")
        assert log_file.suffix == ".log"

    def test_log_file_created_even_for_empty_graph(
        self,
        run_loop: RunLoop,
        exec_config: ExecutionConfig,
        tmp_path: Path,
    ) -> None:
        run_loop.run(exec_config, "main")

        log_file = _find_log_file(tmp_path)
        assert log_file is not None
        assert log_file.exists()


# ── Tests: log lines after dispatch + completion ──────────────────────────────


class TestLogLinesDispatchAndCompletion:
    def test_dispatch_line_written(
        self,
        run_loop: RunLoop,
        graph: MikadoGraph,
        exec_config: ExecutionConfig,
        tmp_path: Path,
    ) -> None:
        root = graph.add_node("root")
        leaf = graph.add_node("dispatch-target", parent_id=root.id)
        run_loop.run(exec_config, "main")

        log_file = _find_log_file(tmp_path)
        assert log_file is not None
        assert f"node_dispatched node_id={leaf.id}" in log_file.read_text()

    def test_completion_line_written_on_success(
        self,
        run_loop: RunLoop,
        graph: MikadoGraph,
        exec_config: ExecutionConfig,
        tmp_path: Path,
    ) -> None:
        root = graph.add_node("root")
        leaf = graph.add_node("success-node", parent_id=root.id)
        run_loop.run(exec_config, "main")

        log_file = _find_log_file(tmp_path)
        assert log_file is not None
        assert f"node_completed node_id={leaf.id}" in log_file.read_text()

    def test_failure_line_written_on_failed_run(
        self,
        graph: MikadoGraph,
        exec_config: ExecutionConfig,
        tmp_path: Path,
    ) -> None:
        ralph = _FakeRalph()
        ralph._success["run-1"] = False
        loop = _make_run_loop(graph, ralph)

        root = graph.add_node("root")
        leaf = graph.add_node("fail-node", parent_id=root.id)
        loop.run(exec_config, "main")

        log_file = _find_log_file(tmp_path)
        assert log_file is not None
        assert f"node_failed node_id={leaf.id}" in log_file.read_text()

    def test_run_started_logged(
        self,
        run_loop: RunLoop,
        graph: MikadoGraph,
        exec_config: ExecutionConfig,
        tmp_path: Path,
    ) -> None:
        graph.add_node("any-node")
        run_loop.run(exec_config, "main")

        log_file = _find_log_file(tmp_path)
        assert log_file is not None
        assert "Run started" in log_file.read_text()

    def test_run_finished_summary_logged(
        self,
        run_loop: RunLoop,
        graph: MikadoGraph,
        exec_config: ExecutionConfig,
        tmp_path: Path,
    ) -> None:
        graph.add_node("any-node")
        run_loop.run(exec_config, "main")

        log_file = _find_log_file(tmp_path)
        assert log_file is not None
        assert "FINAL_TELEMETRY {" in log_file.read_text()

    def test_dispatch_before_completion_in_log(
        self,
        run_loop: RunLoop,
        graph: MikadoGraph,
        exec_config: ExecutionConfig,
        tmp_path: Path,
    ) -> None:
        root = graph.add_node("root")
        leaf = graph.add_node("ordered-node", parent_id=root.id)
        run_loop.run(exec_config, "main")

        log_file = _find_log_file(tmp_path)
        assert log_file is not None
        content = log_file.read_text()
        dispatch_pos = content.find(f"node_dispatched node_id={leaf.id}")
        completion_pos = content.find(f"node_completed node_id={leaf.id}")
        assert dispatch_pos != -1
        assert completion_pos != -1
        assert dispatch_pos < completion_pos


# ── Tests: file closes on KeyboardInterrupt without truncation ────────────────


class TestLogCloseOnKeyboardInterrupt:
    def test_keyboard_interrupt_reraises(
        self,
        graph: MikadoGraph,
        exec_config: ExecutionConfig,
        tmp_path: Path,
    ) -> None:
        ralph = _FakeRalph()
        ralph.raise_interrupt = True
        loop = _make_run_loop(graph, ralph)

        root = graph.add_node("root")
        graph.add_node("interrupted-node", parent_id=root.id)
        with pytest.raises(KeyboardInterrupt):
            loop.run(exec_config, "main")

    def test_log_file_readable_after_interrupt(
        self,
        graph: MikadoGraph,
        exec_config: ExecutionConfig,
        tmp_path: Path,
    ) -> None:
        ralph = _FakeRalph()
        ralph.raise_interrupt = True
        loop = _make_run_loop(graph, ralph)

        root = graph.add_node("root")
        graph.add_node("interrupted-node", parent_id=root.id)
        with pytest.raises(KeyboardInterrupt):
            loop.run(exec_config, "main")

        log_file = _find_log_file(tmp_path)
        assert log_file is not None
        content = log_file.read_text()
        assert len(content) > 0

    def test_interrupted_summary_in_log(
        self,
        graph: MikadoGraph,
        exec_config: ExecutionConfig,
        tmp_path: Path,
    ) -> None:
        ralph = _FakeRalph()
        ralph.raise_interrupt = True
        loop = _make_run_loop(graph, ralph)

        root = graph.add_node("root")
        graph.add_node("interrupted-node", parent_id=root.id)
        with pytest.raises(KeyboardInterrupt):
            loop.run(exec_config, "main")

        log_file = _find_log_file(tmp_path)
        assert log_file is not None
        assert "interrupted" in log_file.read_text().lower()

    def test_dispatch_line_present_before_interrupt(
        self,
        graph: MikadoGraph,
        exec_config: ExecutionConfig,
        tmp_path: Path,
    ) -> None:
        ralph = _FakeRalph()
        ralph.raise_interrupt = True
        loop = _make_run_loop(graph, ralph)

        root = graph.add_node("root")
        leaf = graph.add_node("node-before-interrupt", parent_id=root.id)
        with pytest.raises(KeyboardInterrupt):
            loop.run(exec_config, "main")

        log_file = _find_log_file(tmp_path)
        assert log_file is not None
        assert f"node_dispatched node_id={leaf.id}" in log_file.read_text()

    def test_log_file_closed_after_interrupt(
        self,
        graph: MikadoGraph,
        exec_config: ExecutionConfig,
        tmp_path: Path,
    ) -> None:
        import logging

        ralph = _FakeRalph()
        ralph.raise_interrupt = True
        loop = _make_run_loop(graph, ralph)
        milknado_logger = logging.getLogger("milknado")
        handlers_before = len(milknado_logger.handlers)

        root = graph.add_node("root")
        graph.add_node("interrupted-node", parent_id=root.id)
        with pytest.raises(KeyboardInterrupt):
            loop.run(exec_config, "main")

        assert len(milknado_logger.handlers) == handlers_before
