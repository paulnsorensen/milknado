from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from milknado.domains.graph.graph import MikadoGraph
from milknado.domains.planning.context import build_planning_context
from milknado.domains.planning.planner import Planner, PlanResult


@pytest.fixture()
def tmp_graph(tmp_path: Path) -> MikadoGraph:
    return MikadoGraph(tmp_path / "test.db")


@pytest.fixture()
def mock_crg() -> MagicMock:
    crg = MagicMock()
    crg.get_architecture_overview.return_value = {
        "communities": ["auth", "payments"],
        "entry_points": ["main.py"],
    }
    crg.get_impact_radius.return_value = {
        "files": ["auth.py", "models.py"],
    }
    return crg


class TestBuildPlanningContext:
    def test_includes_goal(
        self, tmp_graph: MikadoGraph, mock_crg: MagicMock
    ) -> None:
        ctx = build_planning_context("extract auth", mock_crg, tmp_graph)
        assert "# Goal" in ctx
        assert "extract auth" in ctx

    def test_includes_architecture(
        self, tmp_graph: MikadoGraph, mock_crg: MagicMock
    ) -> None:
        ctx = build_planning_context("goal", mock_crg, tmp_graph)
        assert "Architecture Overview" in ctx
        assert "communities" in ctx
        mock_crg.get_architecture_overview.assert_called_once()

    def test_includes_empty_graph(
        self, tmp_graph: MikadoGraph, mock_crg: MagicMock
    ) -> None:
        ctx = build_planning_context("goal", mock_crg, tmp_graph)
        assert "No existing nodes" in ctx

    def test_includes_existing_nodes(
        self, tmp_graph: MikadoGraph, mock_crg: MagicMock
    ) -> None:
        tmp_graph.add_node("root goal")
        tmp_graph.add_node("child task", parent_id=1)
        ctx = build_planning_context("goal", mock_crg, tmp_graph)
        assert "[1] root goal (pending)" in ctx
        assert "[2] child task (pending)" in ctx

    def test_includes_dependency_edges(
        self, tmp_graph: MikadoGraph, mock_crg: MagicMock
    ) -> None:
        tmp_graph.add_node("root goal")
        tmp_graph.add_node("child a", parent_id=1)
        tmp_graph.add_node("child b", parent_id=1)
        ctx = build_planning_context("goal", mock_crg, tmp_graph)
        assert "deps: [2, 3]" in ctx

    def test_includes_file_ownership(
        self, tmp_graph: MikadoGraph, mock_crg: MagicMock
    ) -> None:
        tmp_graph.add_node("root goal")
        tmp_graph.set_file_ownership(1, ["src/auth.py", "src/models.py"])
        ctx = build_planning_context("goal", mock_crg, tmp_graph)
        assert "src/auth.py" in ctx
        assert "src/models.py" in ctx

    def test_includes_ready_nodes(
        self, tmp_graph: MikadoGraph, mock_crg: MagicMock
    ) -> None:
        tmp_graph.add_node("root goal")
        tmp_graph.add_node("leaf task", parent_id=1)
        ctx = build_planning_context("goal", mock_crg, tmp_graph)
        assert "Ready to Execute" in ctx
        assert "[2] leaf task" in ctx

    def test_no_ready_section_when_all_done(
        self, tmp_graph: MikadoGraph, mock_crg: MagicMock
    ) -> None:
        tmp_graph.add_node("root goal")
        tmp_graph.mark_running(1)
        tmp_graph.mark_done(1)
        ctx = build_planning_context("goal", mock_crg, tmp_graph)
        assert "Ready to Execute" not in ctx

    def test_includes_instructions(
        self, tmp_graph: MikadoGraph, mock_crg: MagicMock
    ) -> None:
        ctx = build_planning_context("goal", mock_crg, tmp_graph)
        assert "milknado add-node" in ctx
        assert "Dependencies" in ctx

    def test_sections_separated(
        self, tmp_graph: MikadoGraph, mock_crg: MagicMock
    ) -> None:
        ctx = build_planning_context("goal", mock_crg, tmp_graph)
        assert ctx.count("# ") >= 4

    def test_fresh_start_instructions(
        self, tmp_graph: MikadoGraph, mock_crg: MagicMock
    ) -> None:
        ctx = build_planning_context("goal", mock_crg, tmp_graph)
        assert "# Instructions\n" in ctx
        assert "resuming" not in ctx
        assert "Decompose the goal" in ctx

    def test_resume_instructions_when_nodes_exist(
        self, tmp_graph: MikadoGraph, mock_crg: MagicMock
    ) -> None:
        tmp_graph.add_node("root goal")
        ctx = build_planning_context("goal", mock_crg, tmp_graph)
        assert "Instructions (resuming)" in ctx
        assert "Do NOT recreate" in ctx
        assert "milknado add-node" in ctx

    def test_progress_summary(
        self, tmp_graph: MikadoGraph, mock_crg: MagicMock
    ) -> None:
        tmp_graph.add_node("root")
        tmp_graph.add_node("done child", parent_id=1)
        tmp_graph.add_node("pending child", parent_id=1)
        tmp_graph.mark_running(2)
        tmp_graph.mark_done(2)
        ctx = build_planning_context("goal", mock_crg, tmp_graph)
        assert "Progress:" in ctx
        assert "3 total" in ctx
        assert "1 done" in ctx
        assert "2 pending" in ctx

    def test_failed_nodes_section(
        self, tmp_graph: MikadoGraph, mock_crg: MagicMock
    ) -> None:
        tmp_graph.add_node("root")
        tmp_graph.add_node("broken task", parent_id=1)
        tmp_graph.mark_running(2)
        tmp_graph.mark_failed(2)
        ctx = build_planning_context("goal", mock_crg, tmp_graph)
        assert "Failed (need re-planning)" in ctx
        assert "[2] broken task" in ctx

    def test_no_failed_section_when_none_failed(
        self, tmp_graph: MikadoGraph, mock_crg: MagicMock
    ) -> None:
        tmp_graph.add_node("root")
        ctx = build_planning_context("goal", mock_crg, tmp_graph)
        assert "Failed" not in ctx


class TestPlanner:
    def test_build_context_delegates(
        self, tmp_graph: MikadoGraph, mock_crg: MagicMock
    ) -> None:
        planner = Planner(tmp_graph, mock_crg, "claude")
        ctx = planner.build_context("my goal")
        assert "my goal" in ctx
        assert "Architecture" in ctx

    @patch("milknado.domains.planning.planner.subprocess.run")
    def test_launch_writes_context_file(
        self,
        mock_run: MagicMock,
        tmp_path: Path,
        tmp_graph: MikadoGraph,
        mock_crg: MagicMock,
    ) -> None:
        mock_run.return_value = MagicMock(returncode=0)
        planner = Planner(tmp_graph, mock_crg, "claude")
        result = planner.launch("my goal", tmp_path)
        assert result.success is True
        assert result.exit_code == 0
        assert result.context_path is not None
        assert result.context_path.exists()
        content = result.context_path.read_text()
        assert "my goal" in content

    @patch("milknado.domains.planning.planner.subprocess.run")
    def test_launch_calls_agent(
        self,
        mock_run: MagicMock,
        tmp_path: Path,
        tmp_graph: MikadoGraph,
        mock_crg: MagicMock,
    ) -> None:
        mock_run.return_value = MagicMock(returncode=0)
        planner = Planner(tmp_graph, mock_crg, "claude", agent_preset="custom")
        planner.launch("my goal", tmp_path)
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "claude"
        assert "--print" not in cmd
        assert len(cmd) == 2
        assert "my goal" in cmd[1]

    @patch("milknado.domains.planning.planner.subprocess.run")
    def test_launch_claude_preset_uses_stdin(
        self,
        mock_run: MagicMock,
        tmp_path: Path,
        tmp_graph: MikadoGraph,
        mock_crg: MagicMock,
    ) -> None:
        mock_run.return_value = MagicMock(returncode=0)
        planner = Planner(
            tmp_graph,
            mock_crg,
            "claude -p --dangerously-skip-permissions",
            agent_preset="claude",
        )
        planner.launch("my goal", tmp_path)
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert cmd[:3] == ["claude", "-p", "--dangerously-skip-permissions"]
        assert cmd[-1] == "-"
        kwargs = mock_run.call_args[1]
        assert kwargs.get("text") is True
        assert "my goal" in str(kwargs.get("input", ""))

    @patch("milknado.domains.planning.planner.subprocess.run")
    def test_launch_failure(
        self,
        mock_run: MagicMock,
        tmp_path: Path,
        tmp_graph: MikadoGraph,
        mock_crg: MagicMock,
    ) -> None:
        mock_run.return_value = MagicMock(returncode=1)
        planner = Planner(tmp_graph, mock_crg, "claude")
        result = planner.launch("my goal", tmp_path)
        assert result.success is False
        assert result.exit_code == 1

    @patch("milknado.domains.planning.planner.subprocess.run")
    def test_custom_agent_command(
        self,
        mock_run: MagicMock,
        tmp_path: Path,
        tmp_graph: MikadoGraph,
        mock_crg: MagicMock,
    ) -> None:
        mock_run.return_value = MagicMock(returncode=0)
        planner = Planner(tmp_graph, mock_crg, "aider --model opus", agent_preset="custom")
        planner.launch("goal", tmp_path)
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "aider"
        assert "--model" in cmd
        assert "opus" in cmd

    @patch("milknado.domains.planning.planner.subprocess.run")
    def test_launch_runs_in_project_root(
        self,
        mock_run: MagicMock,
        tmp_path: Path,
        tmp_graph: MikadoGraph,
        mock_crg: MagicMock,
    ) -> None:
        mock_run.return_value = MagicMock(returncode=0)
        planner = Planner(tmp_graph, mock_crg, "claude")
        planner.launch("goal", tmp_path)
        assert mock_run.call_args[1]["cwd"] == tmp_path


class TestPlanResult:
    def test_success_result(self) -> None:
        result = PlanResult(success=True, exit_code=0)
        assert result.success is True
        assert result.context_path is None

    def test_failure_result(self) -> None:
        result = PlanResult(
            success=False,
            exit_code=42,
            context_path=Path("/tmp/ctx.md"),
        )
        assert result.exit_code == 42
        assert result.context_path == Path("/tmp/ctx.md")
