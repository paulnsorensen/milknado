from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from milknado.domains.graph import MikadoGraph
from milknado.domains.planning.context import build_planning_context
from milknado.domains.planning.manifest import (
    apply_manifest_to_graph,
    parse_manifest_from_output,
)
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


@pytest.fixture(autouse=True)
def mock_tiktoken_encoding() -> None:
    class _FakeEncoding:
        def encode(self, text: str) -> list[int]:
            return [1 for _ in text]

    with patch("tiktoken.get_encoding", return_value=_FakeEncoding()):
        yield


def _build_ctx(
    *,
    goal: str,
    tmp_path: Path,
    tmp_graph: MikadoGraph,
    mock_crg: MagicMock,
) -> str:
    spec_path = tmp_path / "spec.md"
    spec_path.write_text(goal, encoding="utf-8")
    return build_planning_context(
        spec_path=spec_path,
        crg=mock_crg,
        graph=tmp_graph,
        execution_agent="claude -p",
    )


class TestBuildPlanningContext:
    def test_includes_goal(
        self, tmp_path: Path, tmp_graph: MikadoGraph, mock_crg: MagicMock
    ) -> None:
        ctx = _build_ctx(
            goal="extract auth",
            tmp_path=tmp_path,
            tmp_graph=tmp_graph,
            mock_crg=mock_crg,
        )
        assert "# Spec" in ctx
        assert "extract auth" in ctx

    def test_includes_architecture(
        self, tmp_path: Path, tmp_graph: MikadoGraph, mock_crg: MagicMock
    ) -> None:
        ctx = _build_ctx(
            goal="goal",
            tmp_path=tmp_path,
            tmp_graph=tmp_graph,
            mock_crg=mock_crg,
        )
        assert "Architecture Overview" in ctx
        assert "communities" in ctx
        mock_crg.get_architecture_overview.assert_called_once()
        assert "Atom Budget Heuristics" in ctx

    def test_includes_empty_graph(
        self, tmp_path: Path, tmp_graph: MikadoGraph, mock_crg: MagicMock
    ) -> None:
        ctx = _build_ctx(
            goal="goal",
            tmp_path=tmp_path,
            tmp_graph=tmp_graph,
            mock_crg=mock_crg,
        )
        assert "No existing nodes" in ctx

    def test_includes_existing_nodes(
        self, tmp_path: Path, tmp_graph: MikadoGraph, mock_crg: MagicMock
    ) -> None:
        tmp_graph.add_node("root goal")
        tmp_graph.add_node("child task", parent_id=1)
        ctx = _build_ctx(
            goal="goal",
            tmp_path=tmp_path,
            tmp_graph=tmp_graph,
            mock_crg=mock_crg,
        )
        assert "[1] root goal (pending)" in ctx
        assert "[2] child task (pending)" in ctx

    def test_includes_dependency_edges(
        self, tmp_path: Path, tmp_graph: MikadoGraph, mock_crg: MagicMock
    ) -> None:
        tmp_graph.add_node("root goal")
        tmp_graph.add_node("child a", parent_id=1)
        tmp_graph.add_node("child b", parent_id=1)
        ctx = _build_ctx(
            goal="goal",
            tmp_path=tmp_path,
            tmp_graph=tmp_graph,
            mock_crg=mock_crg,
        )
        assert "deps: [2, 3]" in ctx

    def test_includes_file_ownership(
        self, tmp_path: Path, tmp_graph: MikadoGraph, mock_crg: MagicMock
    ) -> None:
        tmp_graph.add_node("root goal")
        tmp_graph.set_file_ownership(1, ["src/auth.py", "src/models.py"])
        ctx = _build_ctx(
            goal="goal",
            tmp_path=tmp_path,
            tmp_graph=tmp_graph,
            mock_crg=mock_crg,
        )
        assert "src/auth.py" in ctx
        assert "src/models.py" in ctx

    def test_includes_ready_nodes(
        self, tmp_path: Path, tmp_graph: MikadoGraph, mock_crg: MagicMock
    ) -> None:
        tmp_graph.add_node("root goal")
        tmp_graph.add_node("leaf task", parent_id=1)
        ctx = _build_ctx(
            goal="goal",
            tmp_path=tmp_path,
            tmp_graph=tmp_graph,
            mock_crg=mock_crg,
        )
        assert "Ready to Execute" in ctx
        assert "[2] leaf task" in ctx

    def test_no_ready_section_when_all_done(
        self, tmp_path: Path, tmp_graph: MikadoGraph, mock_crg: MagicMock
    ) -> None:
        tmp_graph.add_node("root goal")
        tmp_graph.mark_running(1)
        tmp_graph.mark_done(1)
        ctx = _build_ctx(
            goal="goal",
            tmp_path=tmp_path,
            tmp_graph=tmp_graph,
            mock_crg=mock_crg,
        )
        assert "Ready to Execute" not in ctx

    def test_includes_instructions(
        self, tmp_path: Path, tmp_graph: MikadoGraph, mock_crg: MagicMock
    ) -> None:
        ctx = _build_ctx(
            goal="goal",
            tmp_path=tmp_path,
            tmp_graph=tmp_graph,
            mock_crg=mock_crg,
        )
        assert "milknado add-node" in ctx
        assert "Execution agent target" in ctx
        assert "effective_code_budget" in ctx

    def test_sections_separated(
        self, tmp_path: Path, tmp_graph: MikadoGraph, mock_crg: MagicMock
    ) -> None:
        ctx = _build_ctx(
            goal="goal",
            tmp_path=tmp_path,
            tmp_graph=tmp_graph,
            mock_crg=mock_crg,
        )
        assert ctx.count("# ") >= 4

    def test_fresh_start_instructions(
        self, tmp_path: Path, tmp_graph: MikadoGraph, mock_crg: MagicMock
    ) -> None:
        ctx = _build_ctx(
            goal="goal",
            tmp_path=tmp_path,
            tmp_graph=tmp_graph,
            mock_crg=mock_crg,
        )
        assert "# Instructions\n" in ctx
        assert "resuming" not in ctx
        assert "Decompose the spec" in ctx

    def test_resume_instructions_when_nodes_exist(
        self, tmp_path: Path, tmp_graph: MikadoGraph, mock_crg: MagicMock
    ) -> None:
        tmp_graph.add_node("root goal")
        ctx = _build_ctx(
            goal="goal",
            tmp_path=tmp_path,
            tmp_graph=tmp_graph,
            mock_crg=mock_crg,
        )
        assert "Instructions (resuming)" in ctx
        assert "Do NOT recreate" in ctx
        assert "milknado add-node" in ctx

    def test_progress_summary(
        self, tmp_path: Path, tmp_graph: MikadoGraph, mock_crg: MagicMock
    ) -> None:
        tmp_graph.add_node("root")
        tmp_graph.add_node("done child", parent_id=1)
        tmp_graph.add_node("pending child", parent_id=1)
        tmp_graph.mark_running(2)
        tmp_graph.mark_done(2)
        ctx = _build_ctx(
            goal="goal",
            tmp_path=tmp_path,
            tmp_graph=tmp_graph,
            mock_crg=mock_crg,
        )
        assert "Progress:" in ctx
        assert "3 total" in ctx
        assert "1 done" in ctx
        assert "2 pending" in ctx

    def test_failed_nodes_section(
        self, tmp_path: Path, tmp_graph: MikadoGraph, mock_crg: MagicMock
    ) -> None:
        tmp_graph.add_node("root")
        tmp_graph.add_node("broken task", parent_id=1)
        tmp_graph.mark_running(2)
        tmp_graph.mark_failed(2)
        ctx = _build_ctx(
            goal="goal",
            tmp_path=tmp_path,
            tmp_graph=tmp_graph,
            mock_crg=mock_crg,
        )
        assert "Failed (need re-planning)" in ctx
        assert "[2] broken task" in ctx

    def test_no_failed_section_when_none_failed(
        self, tmp_path: Path, tmp_graph: MikadoGraph, mock_crg: MagicMock
    ) -> None:
        tmp_graph.add_node("root")
        ctx = _build_ctx(
            goal="goal",
            tmp_path=tmp_path,
            tmp_graph=tmp_graph,
            mock_crg=mock_crg,
        )
        assert "Failed" not in ctx


class TestPlanner:
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
        spec_path = tmp_path / "spec.md"
        spec_path.write_text("my goal", encoding="utf-8")
        result = planner.launch(spec_path, tmp_path, execution_agent="claude -p")
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
        planner = Planner(tmp_graph, mock_crg, "claude")
        spec_path = tmp_path / "spec.md"
        spec_path.write_text("my goal", encoding="utf-8")
        planner.launch(spec_path, tmp_path, execution_agent="claude -p")
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "claude"
        assert cmd[-1] == "-"

    @patch("milknado.domains.planning.planner.subprocess.run")
    def test_launch_uses_stdin_input(
        self,
        mock_run: MagicMock,
        tmp_path: Path,
        tmp_graph: MikadoGraph,
        mock_crg: MagicMock,
    ) -> None:
        mock_run.return_value = MagicMock(returncode=0)
        planner = Planner(tmp_graph, mock_crg, "claude -p --dangerously-skip-permissions")
        spec_path = tmp_path / "spec.md"
        spec_path.write_text("my goal", encoding="utf-8")
        planner.launch(spec_path, tmp_path, execution_agent="claude -p")
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
        spec_path = tmp_path / "spec.md"
        spec_path.write_text("my goal", encoding="utf-8")
        result = planner.launch(spec_path, tmp_path, execution_agent="claude -p")
        assert result.success is False
        assert result.exit_code == 1

    @patch("milknado.domains.planning.planner.subprocess.run")
    def test_launch_applies_manifest_to_graph(
        self,
        mock_run: MagicMock,
        tmp_path: Path,
        tmp_graph: MikadoGraph,
        mock_crg: MagicMock,
    ) -> None:
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=(
                "```json\n"
                '{"manifest_version":"milknado.plan.v1","atoms":['
                '{"id":"A1","description":"parent","depends_on":["A2"],"files":["a.py"]},'
                '{"id":"A2","description":"child","depends_on":[],"files":["b.py"]}'
                "]}\n```"
            ),
        )
        planner = Planner(tmp_graph, mock_crg, "claude")
        spec_path = tmp_path / "spec.md"
        spec_path.write_text("my goal", encoding="utf-8")
        result = planner.launch(spec_path, tmp_path, execution_agent="claude -p")
        assert result.success is True
        assert result.nodes_created == 2
        nodes = tmp_graph.get_all_nodes()
        assert len(nodes) == 2
        parent = next(n for n in nodes if n.description == "parent")
        child = next(n for n in nodes if n.description == "child")
        child_ids = [n.id for n in tmp_graph.get_children(parent.id)]
        assert child.id in child_ids
        assert tmp_graph.get_file_ownership(parent.id) == ["a.py"]
        assert tmp_graph.get_file_ownership(child.id) == ["b.py"]

    @patch("milknado.domains.planning.planner.subprocess.run")
    def test_custom_agent_command(
        self,
        mock_run: MagicMock,
        tmp_path: Path,
        tmp_graph: MikadoGraph,
        mock_crg: MagicMock,
    ) -> None:
        mock_run.return_value = MagicMock(returncode=0)
        planner = Planner(tmp_graph, mock_crg, "aider --model opus")
        spec_path = tmp_path / "spec.md"
        spec_path.write_text("goal", encoding="utf-8")
        planner.launch(spec_path, tmp_path, execution_agent="claude -p")
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
        spec_path = tmp_path / "spec.md"
        spec_path.write_text("goal", encoding="utf-8")
        planner.launch(spec_path, tmp_path, execution_agent="claude -p")
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


class TestPlanManifest:
    def test_parse_manifest_from_fenced_json(self) -> None:
        output = (
            "```json\n"
            '{"manifest_version":"milknado.plan.v1","atoms":['
            '{"id":"A1","description":"task","depends_on":[],"files":[]}'
            "]}\n```"
        )
        manifest = parse_manifest_from_output(output)
        assert manifest is not None
        assert manifest.manifest_version == "milknado.plan.v1"
        assert len(manifest.atoms) == 1
        assert manifest.atoms[0].id == "A1"

    def test_apply_manifest_to_graph(self, tmp_graph: MikadoGraph) -> None:
        manifest = parse_manifest_from_output(
            '{"manifest_version":"milknado.plan.v1","atoms":['
            '{"id":"A1","description":"root","depends_on":["A2"],"files":["x.py"]},'
            '{"id":"A2","description":"leaf","depends_on":[],"files":["y.py"]}'
            "]}"
        )
        assert manifest is not None
        created = apply_manifest_to_graph(tmp_graph, manifest)
        assert len(created) == 2
        nodes = tmp_graph.get_all_nodes()
        root = next(n for n in nodes if n.description == "root")
        leaf = next(n for n in nodes if n.description == "leaf")
        assert leaf.id in [n.id for n in tmp_graph.get_children(root.id)]

    def test_apply_manifest_depends_on_makes_prerequisite_ready_first(
        self, tmp_graph: MikadoGraph,
    ) -> None:
        """If root depends_on leaf, only leaf is ready until leaf is done."""
        manifest = parse_manifest_from_output(
            '{"manifest_version":"milknado.plan.v1","atoms":['
            '{"id":"A1","description":"root","depends_on":["A2"],"files":[]},'
            '{"id":"A2","description":"leaf","depends_on":[],"files":[]}'
            "]}"
        )
        assert manifest is not None
        apply_manifest_to_graph(tmp_graph, manifest)
        nodes = tmp_graph.get_all_nodes()
        root = next(n for n in nodes if n.description == "root")
        leaf = next(n for n in nodes if n.description == "leaf")
        ready_ids = {n.id for n in tmp_graph.get_ready_nodes()}
        assert leaf.id in ready_ids
        assert root.id not in ready_ids
        tmp_graph.mark_running(leaf.id)
        tmp_graph.mark_done(leaf.id)
        ready_after = {n.id for n in tmp_graph.get_ready_nodes()}
        assert root.id in ready_after
