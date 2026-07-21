from __future__ import annotations

import dataclasses
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import typer

from milknado.cli.plan import (
    CriticVerdict,
    _exec_plan_non_interactive,
    _parse_critic_output,
    _PlanningSubprocess,
    _run_plan_with_critic,
    run_plan_critic,
)
from milknado.domains.batching import Batch, BatchPlan, FileChange, NewRelationship, SymbolRef
from milknado.domains.batching.change import ChangeDependency, HashAnchors
from milknado.domains.common.config import default_config
from milknado.domains.common.types import DegradationMarker, TilthMap
from milknado.domains.graph import MikadoGraph
from milknado.domains.planning.context import build_planning_context
from milknado.domains.planning.manifest import (
    MANIFEST_VERSION,
    PlanChangeManifest,
    parse_manifest_from_dict,
    parse_manifest_from_output,
)
from milknado.domains.planning.planner import Planner, PlanResult
from milknado.domains.planning.ports import PlanningPorts, PlanningProcessResult


def _ports() -> PlanningPorts:
    tilth = MagicMock()
    tilth.structural_map.return_value = DegradationMarker(source="tilth", reason="mocked")
    return PlanningPorts(tilth=tilth, process=_PlanningSubprocess())


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
    crg.list_communities.return_value = [{"name": f"community_{i}"} for i in range(10)]
    crg.list_flows.return_value = [{"name": f"flow_{i}"} for i in range(10)]
    crg.get_bridge_nodes.return_value = [{"name": f"bridge_{i}"} for i in range(10)]
    crg.get_hub_nodes.return_value = [{"name": f"hub_{i}"} for i in range(10)]
    return crg


class TestPlanningSubprocess:
    def test_local_planner_requires_explicit_capability(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("MILKNADO_LOCAL_PLANNER", raising=False)
        context = tmp_path / "context.md"
        context.write_text("plan context", encoding="utf-8")

        with pytest.raises(ValueError, match="requires MILKNADO_LOCAL_PLANNER"):
            _PlanningSubprocess().run_agent(context, "local-planner", tmp_path)

    def test_local_planner_must_be_inside_project(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        project = tmp_path / "project"
        project.mkdir()
        context = project / "context.md"
        context.write_text("plan context", encoding="utf-8")
        outside = tmp_path / "planner.py"
        outside.write_text("print('no')", encoding="utf-8")
        monkeypatch.setenv("MILKNADO_LOCAL_PLANNER", str(outside))

        with pytest.raises(ValueError, match="within the project root"):
            _PlanningSubprocess().run_agent(context, "local-planner", project)

    @patch("milknado.app.plan.subprocess.run")
    def test_local_planner_uses_exact_interpreter_contract(
        self,
        mock_run: MagicMock,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        context = tmp_path / "context.md"
        context.write_text("plan context", encoding="utf-8")
        planner = tmp_path / "planner.py"
        planner.write_text("print('ok')", encoding="utf-8")
        monkeypatch.setenv("MILKNADO_LOCAL_PLANNER", str(planner))
        mock_run.return_value = MagicMock(returncode=0, stdout="manifest", stderr="")

        result = _PlanningSubprocess().run_agent(context, "local-planner", tmp_path)

        assert result == PlanningProcessResult(exit_code=0, stdout="manifest", stderr="")
        mock_run.assert_called_once()
        assert mock_run.call_args.args[0] == [sys.executable, str(planner)]
        assert mock_run.call_args.kwargs == {
            "cwd": tmp_path,
            "check": False,
            "input": "plan context",
            "text": True,
            "stderr": subprocess.PIPE,
            "stdout": subprocess.PIPE,
        }

    def test_empty_validation_command_returns_explicit_failure(self, tmp_path: Path) -> None:
        assert _PlanningSubprocess().run_validation("   ", {}, tmp_path) == (
            PlanningProcessResult(
                exit_code=1,
                stderr="empty planning validation command",
            )
        )

    @patch("milknado.app.plan.subprocess.run")
    def test_rejects_path_spoof_before_subprocess(
        self,
        mock_run: MagicMock,
        tmp_path: Path,
    ) -> None:
        context = tmp_path / "context.md"
        context.write_text("plan context", encoding="utf-8")

        with pytest.raises(ValueError, match="worker_cmd"):
            _PlanningSubprocess().run_agent(context, "/tmp/claude -p", tmp_path)

        mock_run.assert_not_called()


class TestBuildPlanningContext:
    def test_includes_goal(self, tmp_graph: MikadoGraph, mock_crg: MagicMock) -> None:
        ctx = build_planning_context("extract auth", mock_crg, tmp_graph)
        assert "# Goal" in ctx
        assert "extract auth" in ctx

    def test_includes_compact_crg_block(self, tmp_graph: MikadoGraph, mock_crg: MagicMock) -> None:
        ctx = build_planning_context("goal", mock_crg, tmp_graph)
        assert "Architecture (compact)" in ctx
        # top-5 communities only
        for i in range(5):
            assert f"community_{i}" in ctx
        assert "community_5" not in ctx
        # top-3 flows only
        for i in range(3):
            assert f"flow_{i}" in ctx
        assert "flow_3" not in ctx
        # top-5 bridges
        for i in range(5):
            assert f"bridge_{i}" in ctx
        assert "bridge_5" not in ctx
        # top-5 hubs
        for i in range(5):
            assert f"hub_{i}" in ctx
        assert "hub_5" not in ctx

    def test_no_full_json_architecture_dump(
        self, tmp_graph: MikadoGraph, mock_crg: MagicMock
    ) -> None:
        ctx = build_planning_context("goal", mock_crg, tmp_graph)
        assert "Architecture Overview" not in ctx
        mock_crg.get_architecture_overview.assert_not_called()

    def test_includes_empty_graph(self, tmp_graph: MikadoGraph, mock_crg: MagicMock) -> None:
        ctx = build_planning_context("goal", mock_crg, tmp_graph)
        assert "No existing nodes" in ctx

    def test_includes_existing_nodes(self, tmp_graph: MikadoGraph, mock_crg: MagicMock) -> None:
        tmp_graph.add_node("root goal")
        tmp_graph.add_node("child task", parent_id=1)
        ctx = build_planning_context("goal", mock_crg, tmp_graph)
        assert "[1] root goal (pending)" in ctx
        assert "[2] child task (pending)" in ctx

    def test_includes_dependency_edges(self, tmp_graph: MikadoGraph, mock_crg: MagicMock) -> None:
        tmp_graph.add_node("root goal")
        tmp_graph.add_node("child a", parent_id=1)
        tmp_graph.add_node("child b", parent_id=1)
        ctx = build_planning_context("goal", mock_crg, tmp_graph)
        assert "deps: [2, 3]" in ctx

    def test_includes_file_ownership(self, tmp_graph: MikadoGraph, mock_crg: MagicMock) -> None:
        tmp_graph.add_node("root goal")
        tmp_graph.set_file_ownership(1, ["src/auth.py", "src/models.py"])
        ctx = build_planning_context("goal", mock_crg, tmp_graph)
        assert "src/auth.py" in ctx
        assert "src/models.py" in ctx

    def test_includes_ready_nodes(self, tmp_graph: MikadoGraph, mock_crg: MagicMock) -> None:
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

    def test_includes_instructions(self, tmp_graph: MikadoGraph, mock_crg: MagicMock) -> None:
        ctx = build_planning_context("goal", mock_crg, tmp_graph)
        assert "# Instructions" in ctx
        assert "manifest_version" in ctx

    def test_fresh_start_instructions(self, tmp_graph: MikadoGraph, mock_crg: MagicMock) -> None:
        ctx = build_planning_context("goal", mock_crg, tmp_graph)
        assert "# Instructions\n" in ctx
        assert "Decompose the goal" in ctx

    def test_resume_instructions_when_nodes_exist(
        self, tmp_graph: MikadoGraph, mock_crg: MagicMock
    ) -> None:
        tmp_graph.add_node("root goal")
        ctx = build_planning_context("goal", mock_crg, tmp_graph)
        assert "Instructions (resuming)" in ctx
        assert "Do NOT recreate" in ctx

    def test_instructions_document_preview_then_commit_rhythm(
        self, tmp_graph: MikadoGraph, mock_crg: MagicMock
    ) -> None:
        """Fresh instructions teach plan_batches (preview) -> plan_apply (commit)."""
        ctx = build_planning_context("goal", mock_crg, tmp_graph)
        assert "milknado_plan_batches" in ctx
        assert "milknado_plan_apply" in ctx
        assert "preview" in ctx.lower()
        assert "commit" in ctx.lower()

    def test_resume_instructions_carry_rhythm(
        self, tmp_graph: MikadoGraph, mock_crg: MagicMock
    ) -> None:
        """The resuming branch carries the same preview/commit guidance."""
        tmp_graph.add_node("root goal")
        ctx = build_planning_context("goal", mock_crg, tmp_graph)
        assert "milknado_plan_batches" in ctx
        assert "milknado_plan_apply" in ctx

    def test_progress_summary(self, tmp_graph: MikadoGraph, mock_crg: MagicMock) -> None:
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

    def test_failed_nodes_section(self, tmp_graph: MikadoGraph, mock_crg: MagicMock) -> None:
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

    # --- v2 spec_text tests ---

    def test_spec_text_kwarg_accepted(self, tmp_graph: MikadoGraph, mock_crg: MagicMock) -> None:
        ctx = build_planning_context("goal", mock_crg, tmp_graph, spec_text="my spec body")
        assert ctx  # no error

    def test_spec_section_present_when_spec_text_given(
        self, tmp_graph: MikadoGraph, mock_crg: MagicMock
    ) -> None:
        ctx = build_planning_context(
            "goal", mock_crg, tmp_graph, spec_text="## Overview\nDo the thing"
        )
        assert "# Spec" in ctx
        assert "## Overview" in ctx
        assert "Do the thing" in ctx

    def test_spec_section_absent_when_spec_text_none(
        self, tmp_graph: MikadoGraph, mock_crg: MagicMock
    ) -> None:
        ctx = build_planning_context("goal", mock_crg, tmp_graph, spec_text=None)
        assert "# Spec" not in ctx

    def test_empty_spec_text_raises(self, tmp_graph: MikadoGraph, mock_crg: MagicMock) -> None:
        with pytest.raises(ValueError, match="spec_text"):
            build_planning_context("goal", mock_crg, tmp_graph, spec_text="")

    # --- structural section tests ---

    def test_tilth_kwarg_accepted(self, tmp_graph: MikadoGraph, mock_crg: MagicMock) -> None:
        mock_tilth = MagicMock()
        mock_tilth.structural_map.return_value = TilthMap(
            scope=Path("."), budget_tokens=2000, data={"modules": 5}
        )
        ctx = build_planning_context("goal", mock_crg, tmp_graph, tilth=mock_tilth)
        assert ctx  # no error

    def test_structural_section_with_tilth_map(
        self, tmp_graph: MikadoGraph, mock_crg: MagicMock
    ) -> None:
        mock_tilth = MagicMock()
        mock_tilth.structural_map.return_value = TilthMap(
            scope=Path("."), budget_tokens=2000, data={"modules": 42, "files": 7}
        )
        ctx = build_planning_context("goal", mock_crg, tmp_graph, tilth=mock_tilth)
        assert "Structural Map" in ctx
        assert "modules" in ctx
        assert "42" in ctx

    def test_structural_section_fallback_on_degradation(
        self, tmp_graph: MikadoGraph, mock_crg: MagicMock
    ) -> None:
        mock_tilth = MagicMock()
        mock_tilth.structural_map.return_value = DegradationMarker(
            source="tilth", reason="binary not found"
        )
        ctx = build_planning_context("goal", mock_crg, tmp_graph, tilth=mock_tilth)
        assert "Structural Map" in ctx
        assert "tilth" in ctx
        assert "binary not found" in ctx

    def test_structural_section_fallback_when_tilth_none(
        self, tmp_graph: MikadoGraph, mock_crg: MagicMock
    ) -> None:
        ctx = build_planning_context("goal", mock_crg, tmp_graph, tilth=None)
        assert "Structural Map" in ctx
        assert "not available" in ctx

    # --- v2 prompt schema tests ---

    def test_v2_instructions_contain_manifest_version(
        self, tmp_graph: MikadoGraph, mock_crg: MagicMock
    ) -> None:
        ctx = build_planning_context("goal", mock_crg, tmp_graph)
        assert '"manifest_version": "milknado.plan.v2"' in ctx

    def test_v2_instructions_contain_goal_summary(
        self, tmp_graph: MikadoGraph, mock_crg: MagicMock
    ) -> None:
        ctx = build_planning_context("goal", mock_crg, tmp_graph)
        assert "goal_summary" in ctx

    def test_v2_instructions_contain_hash_anchor_schema(
        self, tmp_graph: MikadoGraph, mock_crg: MagicMock
    ) -> None:
        ctx = build_planning_context("goal", mock_crg, tmp_graph)
        assert "hash_anchors" in ctx
        assert "dependencies" in ctx

    def test_v2_instructions_contain_spec_path(
        self, tmp_graph: MikadoGraph, mock_crg: MagicMock
    ) -> None:
        ctx = build_planning_context("goal", mock_crg, tmp_graph)
        assert "spec_path" in ctx

    def test_v2_instructions_reference_description_and_why(
        self, tmp_graph: MikadoGraph, mock_crg: MagicMock
    ) -> None:
        ctx = build_planning_context("goal", mock_crg, tmp_graph)
        assert "description" in ctx
        assert "why" in ctx.lower()

    def test_v2_instructions_contain_edge_storage_clarification(
        self, tmp_graph: MikadoGraph, mock_crg: MagicMock
    ) -> None:
        ctx = build_planning_context("goal", mock_crg, tmp_graph)
        assert "parent_id" in ctx
        assert "child_id" in ctx

    def test_v2_instructions_resume_also_has_edge_storage(
        self, tmp_graph: MikadoGraph, mock_crg: MagicMock
    ) -> None:
        tmp_graph.add_node("root")
        ctx = build_planning_context("goal", mock_crg, tmp_graph)
        assert "parent_id" in ctx
        assert "child_id" in ctx

    # --- node description truncation tests ---

    def test_node_description_truncated_to_first_line(
        self, tmp_graph: MikadoGraph, mock_crg: MagicMock
    ) -> None:
        tmp_graph.add_node("First line\n\nSecond paragraph with more detail.")
        ctx = build_planning_context("goal", mock_crg, tmp_graph)
        assert "First line" in ctx
        assert "Second paragraph" not in ctx
        assert "\u2026" in ctx  # ellipsis appended

    def test_single_line_description_not_truncated(
        self, tmp_graph: MikadoGraph, mock_crg: MagicMock
    ) -> None:
        tmp_graph.add_node("Only one line here")
        ctx = build_planning_context("goal", mock_crg, tmp_graph)
        assert "Only one line here" in ctx
        # no ellipsis for single-line descriptions
        assert "[1] Only one line here \u2026" not in ctx

    # --- sections count ---

    def test_sections_separated(self, tmp_graph: MikadoGraph, mock_crg: MagicMock) -> None:
        ctx = build_planning_context("goal", mock_crg, tmp_graph)
        assert ctx.count("# ") >= 5  # goal, arch, structural, graph, batching, instructions


class TestPlanner:
    @patch("milknado.app.plan.subprocess.run")
    def test_launch_writes_context_file(
        self,
        mock_run: MagicMock,
        tmp_path: Path,
        tmp_graph: MikadoGraph,
        mock_crg: MagicMock,
    ) -> None:
        mock_run.return_value = MagicMock(returncode=0, stdout="")
        planner = Planner(tmp_graph, mock_crg, "claude", _ports())
        result = planner.launch("my goal", tmp_path)
        assert result.success is True
        assert result.exit_code == 0
        assert result.context_path is not None
        assert result.context_path.exists()
        content = result.context_path.read_text()
        assert "my goal" in content

    @patch("milknado.app.plan.subprocess.run")
    def test_launch_calls_agent(
        self,
        mock_run: MagicMock,
        tmp_path: Path,
        tmp_graph: MikadoGraph,
        mock_crg: MagicMock,
    ) -> None:
        mock_run.return_value = MagicMock(returncode=0, stdout="")
        planner = Planner(tmp_graph, mock_crg, "claude", _ports())
        planner.launch("my goal", tmp_path)
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "claude"
        assert cmd[-1] == "-"

    @patch("milknado.app.plan.subprocess.run")
    def test_launch_uses_stdin_input(
        self,
        mock_run: MagicMock,
        tmp_path: Path,
        tmp_graph: MikadoGraph,
        mock_crg: MagicMock,
    ) -> None:
        mock_run.return_value = MagicMock(returncode=0, stdout="")
        planner = Planner(
            tmp_graph,
            mock_crg,
            "claude -p --dangerously-skip-permissions",
            _ports(),
        )
        planner.launch("my goal", tmp_path)
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert cmd == [
            "claude",
            "-p",
            "--permission-mode",
            "plan",
            "--allowedTools",
            "Read,Glob,Grep",
            "-",
        ]
        kwargs = mock_run.call_args[1]
        assert kwargs.get("text") is True
        assert kwargs.get("input") is not None
        assert "my goal" in str(kwargs.get("input", ""))

    def test_launch_uses_injected_planning_ports(
        self,
        tmp_graph: MikadoGraph,
        mock_crg: MagicMock,
        tmp_path: Path,
    ) -> None:
        process = MagicMock()
        process.run_agent.return_value = PlanningProcessResult(exit_code=0, stdout="")
        tilth = MagicMock()
        tilth.structural_map.return_value = DegradationMarker(
            source="tilth",
            reason="test",
        )
        planner = Planner(
            tmp_graph,
            mock_crg,
            "claude",
            PlanningPorts(tilth=tilth, process=process),
        )

        result = planner.launch("goal", tmp_path)

        assert result.success is True
        process.run_agent.assert_called_once_with(
            tmp_path / ".milknado" / "planning-context.md",
            "claude",
            tmp_path,
        )

    @patch("milknado.app.plan.subprocess.run")
    def test_launch_failure(
        self,
        mock_run: MagicMock,
        tmp_path: Path,
        tmp_graph: MikadoGraph,
        mock_crg: MagicMock,
    ) -> None:
        mock_run.return_value = MagicMock(returncode=1, stdout="")
        planner = Planner(tmp_graph, mock_crg, "claude", _ports())
        result = planner.launch("my goal", tmp_path)
        assert result.success is False
        assert result.exit_code == 1

    @patch("milknado.domains.planning.planner.run_batching")
    @patch("milknado.app.plan.subprocess.run")
    def test_launch_propagates_mega_batch_change_count(
        self,
        mock_run: MagicMock,
        mock_run_batching: MagicMock,
        tmp_path: Path,
        tmp_graph: MikadoGraph,
        mock_crg: MagicMock,
    ) -> None:
        """launch must flow plan.mega_batch_change_count into PlanResult.

        Without this the CLI can never report a mega-batch — a regression to None
        here would silently disable the new CLI surface.
        """
        change_ids = tuple(f"c{i}" for i in range(6))
        stdout = _make_v2_manifest_stdout(
            [
                {"id": cid, "path": f"src/f{i}.py", "description": f"Change {i}"}
                for i, cid in enumerate(change_ids)
            ]
        )
        mock_run.return_value = MagicMock(returncode=0, stdout=stdout)
        mock_run_batching.return_value = BatchPlan(
            batches=(Batch(index=0, change_ids=change_ids, depends_on=()),),
            spread_report=(),
            solver_status="OPTIMAL",
        )
        planner = Planner(tmp_graph, mock_crg, "claude", _ports())
        result = planner.launch("goal", tmp_path)
        assert result.mega_batch_change_count == 6

    @patch("milknado.domains.planning.planner.run_batching")
    @patch("milknado.app.plan.subprocess.run")
    def test_launch_mega_batch_change_count_none_when_within_threshold(
        self,
        mock_run: MagicMock,
        mock_run_batching: MagicMock,
        tmp_path: Path,
        tmp_graph: MikadoGraph,
        mock_crg: MagicMock,
    ) -> None:
        """A plan with no offending batch leaves mega_batch_change_count None."""
        stdout = _make_v2_manifest_stdout(
            [{"id": "c0", "path": "src/f0.py", "description": "Change 0"}]
        )
        mock_run.return_value = MagicMock(returncode=0, stdout=stdout)
        mock_run_batching.return_value = BatchPlan(
            batches=(Batch(index=0, change_ids=("c0",), depends_on=()),),
            spread_report=(),
            solver_status="OPTIMAL",
        )
        planner = Planner(tmp_graph, mock_crg, "claude", _ports())
        result = planner.launch("goal", tmp_path)
        assert result.mega_batch_change_count is None

    @patch("milknado.app.plan.subprocess.run")
    def test_custom_agent_command(
        self,
        mock_run: MagicMock,
        tmp_path: Path,
        tmp_graph: MikadoGraph,
        mock_crg: MagicMock,
    ) -> None:
        planner = Planner(tmp_graph, mock_crg, "aider --model opus", _ports())
        with pytest.raises(
            ValueError,
            match=r"^unsupported planning agent executable: 'aider'$",
        ):
            planner.launch("goal", tmp_path)
        mock_run.assert_not_called()

    @patch("milknado.app.plan.subprocess.run")
    def test_launch_runs_in_project_root(
        self,
        mock_run: MagicMock,
        tmp_path: Path,
        tmp_graph: MikadoGraph,
        mock_crg: MagicMock,
    ) -> None:
        mock_run.return_value = MagicMock(returncode=0, stdout="")
        planner = Planner(tmp_graph, mock_crg, "claude", _ports())
        planner.launch("goal", tmp_path)
        assert mock_run.call_args[1]["cwd"] == tmp_path

    @patch("milknado.app.plan.subprocess.run")
    def test_launch_excludes_untrusted_repo_mcp_config(
        self,
        mock_run: MagicMock,
        tmp_path: Path,
        tmp_graph: MikadoGraph,
        mock_crg: MagicMock,
    ) -> None:
        (tmp_path / ".mcp.json").write_text('{"mcpServers": {}}', encoding="utf-8")
        mock_run.return_value = MagicMock(returncode=0, stdout="")
        planner = Planner(tmp_graph, mock_crg, "claude", _ports())
        planner.launch("goal", tmp_path)
        cmd = mock_run.call_args[0][0]
        assert "--mcp-config" not in cmd
        assert str(tmp_path / ".mcp.json") not in cmd

    @patch("milknado.app.plan.subprocess.run")
    def test_validation_hook_accepts_manifest(
        self,
        mock_run: MagicMock,
        tmp_path: Path,
        tmp_graph: MikadoGraph,
        mock_crg: MagicMock,
    ) -> None:
        stdout = _make_v2_manifest_stdout(
            [{"id": "c1", "path": "src/foo.py", "description": "Add Foo"}]
        )

        def _side_effect(argv: list[str], *args: object, **kwargs: object) -> MagicMock:
            if argv and argv[0] == "python":
                return MagicMock(returncode=0, stdout="", stderr="")
            return MagicMock(returncode=0, stdout=stdout)

        mock_run.side_effect = _side_effect
        planner = Planner(
            tmp_graph,
            mock_crg,
            "claude",
            _ports(),
            planning_validation_hook="python scripts/validate_plan.py",
        )
        result = planner.launch("goal", tmp_path)
        assert result.solver_status in {"OPTIMAL", "FEASIBLE", "UNKNOWN"}
        assert result.change_count == 1
        hook_call = next(
            call for call in mock_run.call_args_list if call.args and call.args[0][0] == "python"
        )
        assert hook_call.args[0][:2] == ["python", "scripts/validate_plan.py"]
        hook_payload = hook_call.kwargs["input"]
        assert '"manifest_version": "milknado.plan.v2"' in hook_payload
        assert '"id": "c1"' in hook_payload

    @patch("milknado.app.plan.subprocess.run")
    def test_validation_hook_rejects_manifest(
        self,
        mock_run: MagicMock,
        tmp_path: Path,
        tmp_graph: MikadoGraph,
        mock_crg: MagicMock,
    ) -> None:
        stdout = _make_v2_manifest_stdout(
            [{"id": "c1", "path": "src/foo.py", "description": "Add Foo"}]
        )

        def _side_effect(argv: list[str], *args: object, **kwargs: object) -> MagicMock:
            if argv and argv[0] == "node":
                return MagicMock(returncode=1, stdout="", stderr="missing goal summary")
            return MagicMock(returncode=0, stdout=stdout)

        mock_run.side_effect = _side_effect
        planner = Planner(
            tmp_graph,
            mock_crg,
            "claude",
            _ports(),
            planning_validation_hook="node hook.mjs",
        )
        result = planner.launch("goal", tmp_path)
        assert result.success is False
        assert result.exit_code == 1
        assert result.solver_status == "VALIDATION_FAILED"
        assert result.change_count == 1


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

    def test_new_fields_default_to_zero(self) -> None:
        result = PlanResult(success=True, exit_code=0)
        assert result.nodes_created == 0
        assert result.batch_count == 0
        assert result.oversized_count == 0
        assert result.solver_status == ""

    def test_new_fields_can_be_set(self) -> None:
        result = PlanResult(
            success=True,
            exit_code=0,
            nodes_created=5,
            batch_count=3,
            oversized_count=1,
            solver_status="OPTIMAL",
        )
        assert result.nodes_created == 5
        assert result.batch_count == 3
        assert result.oversized_count == 1
        assert result.solver_status == "OPTIMAL"


import json as _json_module  # noqa: E402


def _make_v2_manifest_stdout(changes: list[dict]) -> str:  # type: ignore[type-arg]
    payload = {
        "manifest_version": "milknado.plan.v2",
        "goal": "Test goal",
        "goal_summary": "A short test goal summary for integration.",
        "changes": changes,
    }
    return "```json\n" + _json_module.dumps(payload) + "\n```"


class TestPlannerSpecPath:
    @patch("milknado.app.plan.subprocess.run")
    def test_spec_path_read_and_passed(
        self,
        mock_run: MagicMock,
        tmp_path: Path,
        tmp_graph: MikadoGraph,
        mock_crg: MagicMock,
    ) -> None:
        mock_run.return_value = MagicMock(returncode=0, stdout="")
        spec = tmp_path / "spec.md"
        spec.write_text("## My Spec\nDo the thing.", encoding="utf-8")
        planner = Planner(tmp_graph, mock_crg, "claude", _ports())
        result = planner.launch("goal", tmp_path, spec_path=spec)
        assert result.context_path is not None
        content = result.context_path.read_text()
        assert "My Spec" in content
        assert "Do the thing." in content

    def test_spec_path_missing_raises(
        self,
        tmp_path: Path,
        tmp_graph: MikadoGraph,
        mock_crg: MagicMock,
    ) -> None:
        planner = Planner(tmp_graph, mock_crg, "claude", _ports())
        with pytest.raises(FileNotFoundError, match="spec_path"):
            planner.launch("goal", tmp_path, spec_path=tmp_path / "nonexistent.md")

    @patch("milknado.app.plan.subprocess.run")
    def test_no_manifest_returns_no_manifest_status(
        self,
        mock_run: MagicMock,
        tmp_path: Path,
        tmp_graph: MikadoGraph,
        mock_crg: MagicMock,
    ) -> None:
        mock_run.return_value = MagicMock(returncode=0, stdout="no JSON block here")
        planner = Planner(tmp_graph, mock_crg, "claude", _ports())
        result = planner.launch("goal", tmp_path)
        assert result.solver_status == "NO_MANIFEST"
        assert result.nodes_created == 0
        assert result.batch_count == 0

    @patch("milknado.app.plan.subprocess.run")
    def test_no_manifest_no_telemetry(
        self,
        mock_run: MagicMock,
        tmp_path: Path,
        tmp_graph: MikadoGraph,
        mock_crg: MagicMock,
    ) -> None:
        mock_run.return_value = MagicMock(returncode=0, stdout="no block")
        planner = Planner(tmp_graph, mock_crg, "claude", _ports())
        planner.launch("goal", tmp_path)
        assert not (tmp_path / ".milknado" / "calibration.jsonl").exists()

    @patch("milknado.app.plan.subprocess.run")
    def test_happy_path_creates_nodes_and_telemetry(
        self,
        mock_run: MagicMock,
        tmp_path: Path,
        tmp_graph: MikadoGraph,
        mock_crg: MagicMock,
    ) -> None:
        stdout = _make_v2_manifest_stdout(
            [
                {"id": "c1", "path": "src/foo.py", "description": "Add Foo class"},
            ]
        )
        mock_run.return_value = MagicMock(returncode=0, stdout=stdout)
        planner = Planner(tmp_graph, mock_crg, "claude", _ports())
        result = planner.launch("Test goal", tmp_path)
        assert result.nodes_created > 0
        assert (tmp_path / ".milknado" / "calibration.jsonl").exists()

    @patch("milknado.app.plan.subprocess.run")
    def test_crg_failure_still_runs_batching(
        self,
        mock_run: MagicMock,
        tmp_path: Path,
        tmp_graph: MikadoGraph,
        mock_crg: MagicMock,
    ) -> None:
        mock_crg.ensure_graph.side_effect = RuntimeError("CRG unavailable")
        stdout = _make_v2_manifest_stdout(
            [
                {"id": "c1", "path": "src/foo.py", "description": "Add Foo"},
            ]
        )
        mock_run.return_value = MagicMock(returncode=0, stdout=stdout)
        planner = Planner(tmp_graph, mock_crg, "claude", _ports())
        result = planner.launch("goal", tmp_path)
        assert result.nodes_created > 0


def _wrap(payload: dict) -> str:
    import json as _json

    return "preamble\n```json\n" + _json.dumps(payload) + "\n```\ntrailer"


class TestParseManifestFromDict:
    """parse_manifest_from_dict validates a raw dict, returning None on failure."""

    def _valid_dict(self) -> dict:
        return {
            "manifest_version": "milknado.plan.v2",
            "goal": "Refactor foo",
            "goal_summary": "Move foo into its own module",
            "changes": [
                {"id": "c1", "path": "src/foo.py", "description": "Add Foo"},
            ],
        }

    def test_valid_dict_returns_manifest(self) -> None:
        manifest = parse_manifest_from_dict(self._valid_dict())
        assert manifest is not None
        assert manifest.manifest_version == MANIFEST_VERSION
        assert manifest.goal == "Refactor foo"
        assert manifest.goal_summary == "Move foo into its own module"
        assert len(manifest.changes) == 1
        assert manifest.changes[0].id == "c1"

    def test_non_dict_returns_none(self) -> None:
        assert parse_manifest_from_dict(["not", "a", "dict"]) is None

    def test_wrong_version_returns_none(self) -> None:
        bad = self._valid_dict()
        bad["manifest_version"] = "milknado.plan.v1"
        assert parse_manifest_from_dict(bad) is None

    def test_missing_goal_returns_none(self) -> None:
        bad = self._valid_dict()
        del bad["goal"]
        assert parse_manifest_from_dict(bad) is None

    def test_invalid_edit_kind_returns_none(self) -> None:
        bad = self._valid_dict()
        bad["changes"][0]["edit_kind"] = "frobnicate"
        assert parse_manifest_from_dict(bad) is None

    def test_text_path_delegates_to_dict_path(self) -> None:
        """The text parser yields the same manifest the dict parser does for the same payload."""
        payload = self._valid_dict()
        from_text = parse_manifest_from_output(_wrap(payload))
        from_dict = parse_manifest_from_dict(payload)
        assert from_text == from_dict
        assert from_text is not None

    def test_missing_goal_summary_returns_none(self) -> None:
        bad = self._valid_dict()
        del bad["goal_summary"]
        assert parse_manifest_from_dict(bad) is None

    def test_blank_goal_summary_returns_none(self) -> None:
        bad = self._valid_dict()
        bad["goal_summary"] = "   "
        assert parse_manifest_from_dict(bad) is None

    def test_goal_and_summary_are_stripped(self) -> None:
        """The moved validation strips surrounding whitespace from goal/goal_summary."""
        payload = self._valid_dict()
        payload["goal"] = "  Refactor foo  "
        payload["goal_summary"] = "\tMove foo into its own module\n"
        manifest = parse_manifest_from_dict(payload)
        assert manifest is not None
        assert manifest.goal == "Refactor foo"
        assert manifest.goal_summary == "Move foo into its own module"

    def test_non_string_spec_path_returns_none(self) -> None:
        bad = self._valid_dict()
        bad["spec_path"] = 42
        assert parse_manifest_from_dict(bad) is None

    def test_valid_spec_path_round_trips(self) -> None:
        payload = self._valid_dict()
        payload["spec_path"] = "specs/foo.md"
        manifest = parse_manifest_from_dict(payload)
        assert manifest is not None
        assert manifest.spec_path == "specs/foo.md"

    def test_valid_new_relationship_round_trips(self) -> None:
        payload = self._valid_dict()
        payload["changes"].append(
            {"id": "c2", "path": "src/bar.py", "description": "Add Bar", "depends_on": ["c1"]}
        )
        payload["new_relationships"] = [
            {"source_change_id": "c1", "dependant_change_id": "c2", "reason": "new_import"}
        ]
        manifest = parse_manifest_from_dict(payload)
        assert manifest is not None
        assert len(manifest.new_relationships) == 1
        rel = manifest.new_relationships[0]
        assert rel.source_change_id == "c1"
        assert rel.dependant_change_id == "c2"
        assert rel.reason == "new_import"

    def test_invalid_relationship_reason_returns_none(self) -> None:
        payload = self._valid_dict()
        payload["changes"].append({"id": "c2", "path": "src/bar.py", "description": "Add Bar"})
        payload["new_relationships"] = [
            {"source_change_id": "c1", "dependant_change_id": "c2", "reason": "telepathy"}
        ]
        assert parse_manifest_from_dict(payload) is None

    def test_relationship_with_unknown_change_id_returns_none(self) -> None:
        payload = self._valid_dict()
        payload["new_relationships"] = [
            {"source_change_id": "c1", "dependant_change_id": "ghost", "reason": "new_import"}
        ]
        assert parse_manifest_from_dict(payload) is None

    def test_duplicate_change_id_returns_none(self) -> None:
        bad = self._valid_dict()
        bad["changes"].append({"id": "c1", "path": "src/dup.py", "description": "Dup"})
        assert parse_manifest_from_dict(bad) is None

    def test_depends_on_unknown_id_returns_none(self) -> None:
        bad = self._valid_dict()
        bad["changes"][0]["depends_on"] = ["nonexistent"]
        assert parse_manifest_from_dict(bad) is None

    def test_absolute_change_path_returns_none(self) -> None:
        bad = self._valid_dict()
        bad["changes"][0]["path"] = "/etc/passwd"
        assert parse_manifest_from_dict(bad) is None

    def test_traversal_change_path_returns_none(self) -> None:
        bad = self._valid_dict()
        bad["changes"][0]["path"] = "../escape.py"
        assert parse_manifest_from_dict(bad) is None

    def test_absolute_symbol_file_returns_none(self) -> None:
        bad = self._valid_dict()
        bad["changes"][0]["symbols"] = [{"name": "Foo", "file": "/etc/passwd"}]
        assert parse_manifest_from_dict(bad) is None

    def test_traversal_symbol_file_returns_none(self) -> None:
        bad = self._valid_dict()
        bad["changes"][0]["symbols"] = [{"name": "Foo", "file": "../secret.py"}]
        assert parse_manifest_from_dict(bad) is None


class TestPlanChangeManifest:
    def test_happy_path_parses_full_manifest(self) -> None:
        output = _wrap(
            {
                "manifest_version": "milknado.plan.v2",
                "goal": "Refactor foo",
                "goal_summary": "Move foo into its own module",
                "changes": [
                    {
                        "id": "c1",
                        "path": "src/foo.py",
                        "edit_kind": "modify",
                        "description": "Update Foo class",
                        "symbols": [{"name": "Foo", "file": "src/foo.py"}],
                        "hash_anchors": {
                            "before": "sha256:foo-before",
                            "after": "sha256:foo-after",
                        },
                        "dependencies": [
                            {
                                "path": "src/baz.py",
                                "symbols": [{"name": "Baz", "file": "src/baz.py"}],
                                "hash_anchors": {
                                    "before": "sha256:baz-before",
                                    "after": "sha256:baz-after",
                                },
                                "reason": "import edge",
                            }
                        ],
                        "depends_on": ["c2"],
                    },
                    {
                        "id": "c2",
                        "path": "src/bar.py",
                        "edit_kind": "add",
                        "description": "Add Bar module",
                    },
                ],
                "new_relationships": [
                    {
                        "source_change_id": "c2",
                        "dependant_change_id": "c1",
                        "reason": "new_import",
                    },
                ],
            }
        )
        manifest = parse_manifest_from_output(output)
        assert manifest is not None
        assert manifest.manifest_version == MANIFEST_VERSION
        assert len(manifest.changes) == 2
        assert manifest.changes[0] == FileChange(
            id="c1",
            path="src/foo.py",
            edit_kind="modify",
            description="Update Foo class",
            symbols=(SymbolRef(name="Foo", file="src/foo.py"),),
            hash_anchors=HashAnchors(before="sha256:foo-before", after="sha256:foo-after"),
            dependencies=(
                ChangeDependency(
                    path="src/baz.py",
                    symbols=(SymbolRef(name="Baz", file="src/baz.py"),),
                    hash_anchors=HashAnchors(before="sha256:baz-before", after="sha256:baz-after"),
                    reason="import edge",
                ),
            ),
            depends_on=("c2",),
        )
        assert manifest.changes[1].edit_kind == "add"
        assert manifest.changes[1].symbols == ()
        assert manifest.new_relationships == (
            NewRelationship(
                source_change_id="c2",
                dependant_change_id="c1",
                reason="new_import",
            ),
        )

    def test_returns_none_when_no_fenced_block(self) -> None:
        assert parse_manifest_from_output("just prose, no block") is None

    def test_returns_none_on_malformed_json(self) -> None:
        assert parse_manifest_from_output("```json\n{not json\n```") is None

    def test_rejects_wrong_manifest_version(self) -> None:
        output = _wrap(
            {
                "manifest_version": "milknado.plan.v1",
                "changes": [],
            }
        )
        assert parse_manifest_from_output(output) is None

    def test_rejects_duplicate_change_ids(self) -> None:
        output = _wrap(
            {
                "manifest_version": "milknado.plan.v2",
                "goal": "Fix bug",
                "goal_summary": "Fix null pointer",
                "changes": [
                    {"id": "c1", "path": "a.py", "description": "Fix a"},
                    {"id": "c1", "path": "b.py", "description": "Fix b"},
                ],
            }
        )
        assert parse_manifest_from_output(output) is None

    def test_rejects_unknown_depends_on(self) -> None:
        output = _wrap(
            {
                "manifest_version": "milknado.plan.v2",
                "goal": "Fix bug",
                "goal_summary": "Fix null pointer",
                "changes": [
                    {"id": "c1", "path": "a.py", "description": "Fix a", "depends_on": ["ghost"]},
                ],
            }
        )
        assert parse_manifest_from_output(output) is None

    def test_rejects_unknown_relationship_endpoint(self) -> None:
        output = _wrap(
            {
                "manifest_version": "milknado.plan.v2",
                "goal": "Fix bug",
                "goal_summary": "Fix null pointer",
                "changes": [{"id": "c1", "path": "a.py", "description": "Fix a"}],
                "new_relationships": [
                    {
                        "source_change_id": "ghost",
                        "dependant_change_id": "c1",
                        "reason": "new_import",
                    },
                ],
            }
        )
        assert parse_manifest_from_output(output) is None

    def test_rejects_invalid_edit_kind(self) -> None:
        output = _wrap(
            {
                "manifest_version": "milknado.plan.v2",
                "goal": "Fix bug",
                "goal_summary": "Fix null pointer",
                "changes": [
                    {"id": "c1", "path": "a.py", "edit_kind": "explode", "description": "Fix a"},
                ],
            }
        )
        assert parse_manifest_from_output(output) is None

    def test_rejects_invalid_relationship_reason(self) -> None:
        output = _wrap(
            {
                "manifest_version": "milknado.plan.v2",
                "goal": "Fix bug",
                "goal_summary": "Fix null pointer",
                "changes": [
                    {"id": "c1", "path": "a.py", "description": "Fix a"},
                    {"id": "c2", "path": "b.py", "description": "Fix b"},
                ],
                "new_relationships": [
                    {
                        "source_change_id": "c1",
                        "dependant_change_id": "c2",
                        "reason": "cosmic_ray",
                    },
                ],
            }
        )
        assert parse_manifest_from_output(output) is None

    def test_defaults_for_optional_fields(self) -> None:
        output = _wrap(
            {
                "manifest_version": "milknado.plan.v2",
                "goal": "Fix bug",
                "goal_summary": "Fix null pointer",
                "changes": [{"id": "c1", "path": "a.py", "description": "Fix a"}],
            }
        )
        manifest = parse_manifest_from_output(output)
        assert manifest is not None
        only = manifest.changes[0]
        assert only.edit_kind == "modify"
        assert only.symbols == ()
        assert only.hash_anchors is None
        assert only.dependencies == ()
        assert only.depends_on == ()
        assert manifest.new_relationships == ()

    def test_parses_hash_anchors_and_dependencies(self) -> None:
        output = _wrap(
            {
                "manifest_version": "milknado.plan.v2",
                "goal": "Refactor auth",
                "goal_summary": "Extract auth slice",
                "changes": [
                    {
                        "id": "c1",
                        "path": "src/foo.py",
                        "description": "Change foo",
                        "hash_anchors": {
                            "before": "sha256:foo-before",
                            "after": "sha256:foo-after",
                        },
                        "dependencies": [
                            {
                                "path": "src/bar.py",
                                "symbols": [{"name": "bar", "file": "src/bar.py"}],
                                "hash_anchors": {
                                    "before": "sha256:bar-before",
                                    "after": "sha256:bar-after",
                                },
                                "reason": "call site",
                            }
                        ],
                    }
                ],
            }
        )
        manifest = parse_manifest_from_output(output)
        assert manifest is not None
        change = manifest.changes[0]
        assert change.hash_anchors == HashAnchors(
            before="sha256:foo-before", after="sha256:foo-after"
        )
        assert change.dependencies == (
            ChangeDependency(
                path="src/bar.py",
                symbols=(SymbolRef(name="bar", file="src/bar.py"),),
                hash_anchors=HashAnchors(before="sha256:bar-before", after="sha256:bar-after"),
                reason="call site",
            ),
        )

    def test_direct_construction_is_frozen(self) -> None:
        manifest = PlanChangeManifest(
            manifest_version=MANIFEST_VERSION,
            goal="some goal",
            goal_summary="a summary",
            spec_path=None,
            changes=(),
            new_relationships=(),
        )
        with pytest.raises(Exception):  # noqa: B017, PT011 — frozen dataclass
            manifest.changes = ()  # ty: ignore[invalid-assignment]

    def test_v2_parse_with_descriptions(self) -> None:
        output = _wrap(
            {
                "manifest_version": "milknado.plan.v2",
                "goal": "Refactor auth module",
                "goal_summary": "Extract auth into its own domain slice",
                "spec_path": "specs/auth-refactor.md",
                "changes": [
                    {
                        "id": "c1",
                        "path": "src/auth.py",
                        "description": "Add AuthService class",
                    },
                    {
                        "id": "c2",
                        "path": "src/main.py",
                        "description": "Wire AuthService into DI container",
                        "depends_on": ["c1"],
                    },
                ],
            }
        )
        manifest = parse_manifest_from_output(output)
        assert manifest is not None
        assert manifest.goal == "Refactor auth module"
        assert manifest.goal_summary == "Extract auth into its own domain slice"
        assert manifest.spec_path == "specs/auth-refactor.md"
        assert manifest.changes[0].description == "Add AuthService class"
        assert manifest.changes[1].description == "Wire AuthService into DI container"

    def test_spec_path_can_be_none(self) -> None:
        output = _wrap(
            {
                "manifest_version": "milknado.plan.v2",
                "goal": "Fix bug",
                "goal_summary": "Fix null pointer in handler",
                "changes": [
                    {"id": "c1", "path": "src/handler.py", "description": "Guard against None"},
                ],
            }
        )
        manifest = parse_manifest_from_output(output)
        assert manifest is not None
        assert manifest.spec_path is None

    def test_spec_path_can_be_a_string(self) -> None:
        output = _wrap(
            {
                "manifest_version": "milknado.plan.v2",
                "goal": "New feature",
                "goal_summary": "Add export functionality",
                "spec_path": "specs/export.md",
                "changes": [
                    {"id": "c1", "path": "src/export.py", "description": "Implement exporter"},
                ],
            }
        )
        manifest = parse_manifest_from_output(output)
        assert manifest is not None
        assert manifest.spec_path == "specs/export.md"

    def test_rejects_empty_goal(self) -> None:
        output = _wrap(
            {
                "manifest_version": "milknado.plan.v2",
                "goal": "",
                "goal_summary": "some summary",
                "changes": [
                    {"id": "c1", "path": "src/foo.py", "description": "Do something"},
                ],
            }
        )
        assert parse_manifest_from_output(output) is None

    def test_rejects_missing_goal(self) -> None:
        output = _wrap(
            {
                "manifest_version": "milknado.plan.v2",
                "goal_summary": "some summary",
                "changes": [
                    {"id": "c1", "path": "src/foo.py", "description": "Do something"},
                ],
            }
        )
        assert parse_manifest_from_output(output) is None

    def test_rejects_empty_goal_summary(self) -> None:
        output = _wrap(
            {
                "manifest_version": "milknado.plan.v2",
                "goal": "Refactor auth",
                "goal_summary": "",
                "changes": [
                    {"id": "c1", "path": "src/foo.py", "description": "Do something"},
                ],
            }
        )
        assert parse_manifest_from_output(output) is None

    def test_rejects_missing_goal_summary(self) -> None:
        output = _wrap(
            {
                "manifest_version": "milknado.plan.v2",
                "goal": "Refactor auth",
                "changes": [
                    {"id": "c1", "path": "src/foo.py", "description": "Do something"},
                ],
            }
        )
        assert parse_manifest_from_output(output) is None

    def test_rejects_empty_description_on_any_change(self) -> None:
        output = _wrap(
            {
                "manifest_version": "milknado.plan.v2",
                "goal": "Refactor auth",
                "goal_summary": "Extract auth slice",
                "changes": [
                    {"id": "c1", "path": "src/foo.py", "description": "Add class"},
                    {"id": "c2", "path": "src/bar.py", "description": ""},
                ],
            }
        )
        assert parse_manifest_from_output(output) is None

    def test_rejects_missing_description_on_any_change(self) -> None:
        output = _wrap(
            {
                "manifest_version": "milknado.plan.v2",
                "goal": "Refactor auth",
                "goal_summary": "Extract auth slice",
                "changes": [
                    {"id": "c1", "path": "src/foo.py"},
                ],
            }
        )
        assert parse_manifest_from_output(output) is None

    def test_rejects_invalid_hash_anchors_shape(self) -> None:
        output = _wrap(
            {
                "manifest_version": "milknado.plan.v2",
                "goal": "Refactor auth",
                "goal_summary": "Extract auth slice",
                "changes": [
                    {
                        "id": "c1",
                        "path": "src/foo.py",
                        "description": "Do something",
                        "hash_anchors": {"before": "", "after": "sha256:after"},
                    }
                ],
            }
        )
        assert parse_manifest_from_output(output) is None

    def test_rejects_invalid_dependency_shape(self) -> None:
        output = _wrap(
            {
                "manifest_version": "milknado.plan.v2",
                "goal": "Refactor auth",
                "goal_summary": "Extract auth slice",
                "changes": [
                    {
                        "id": "c1",
                        "path": "src/foo.py",
                        "description": "Do something",
                        "dependencies": [{"path": "", "reason": "bad dep"}],
                    }
                ],
            }
        )
        assert parse_manifest_from_output(output) is None

    def test_v1_rejection_still_works(self) -> None:
        output = _wrap(
            {
                "manifest_version": "milknado.plan.v1",
                "changes": [],
            }
        )
        assert parse_manifest_from_output(output) is None

    def test_internal_construction_allows_empty_description(self) -> None:
        """Solver-internal construction with default description="" is allowed."""
        change = FileChange(id="c1", path="src/foo.py")
        assert change.description == ""


def _cfg(tmp_path: Path, **overrides: object) -> object:
    return dataclasses.replace(default_config(tmp_path), **overrides)


def _plan_result(**overrides: object) -> PlanResult:
    base = {
        "success": True,
        "exit_code": 0,
        "solver_status": "OPTIMAL",
        "change_count": 1,
        "batch_count": 1,
    }
    base.update(overrides)
    return PlanResult(**base)


class TestParseCriticOutput:
    def test_approve_verdict(self) -> None:
        verdict = _parse_critic_output("<verdict>approve</verdict>")
        assert verdict == CriticVerdict(approved=True, feedback="")

    def test_revise_verdict_with_feedback(self) -> None:
        output = (
            "<verdict>revise</verdict>\n<feedback>split the change into two batches</feedback>"
        )
        verdict = _parse_critic_output(output)
        assert verdict == CriticVerdict(
            approved=False, feedback="split the change into two batches"
        )

    def test_revise_verdict_with_no_feedback_tag(self) -> None:
        verdict = _parse_critic_output("<verdict>revise</verdict>")
        assert verdict == CriticVerdict(approved=False, feedback="")

    def test_unparseable_output_returns_none(self) -> None:
        assert _parse_critic_output("the plan looks fine to me") is None


class TestRunPlanCritic:
    @patch("milknado.app.plan._spawn_plan_critic")
    def test_returns_approved_verdict(self, mock_spawn: MagicMock, tmp_path: Path) -> None:
        mock_spawn.return_value = "<verdict>approve</verdict>"
        cfg = _cfg(tmp_path, plan_reviewer_agent="claude --model sonnet -p")

        verdict = run_plan_critic("goal", "manifest summary", cfg)

        assert verdict == CriticVerdict(approved=True, feedback="")
        mock_spawn.assert_called_once()

    @patch("milknado.app.plan._spawn_plan_critic")
    def test_parses_revise_with_feedback(self, mock_spawn: MagicMock, tmp_path: Path) -> None:
        mock_spawn.return_value = (
            "<verdict>revise</verdict>\n<feedback>batch is too large</feedback>"
        )
        cfg = _cfg(tmp_path, plan_reviewer_agent="claude --model sonnet -p")

        verdict = run_plan_critic("goal", "manifest summary", cfg)

        assert verdict == CriticVerdict(approved=False, feedback="batch is too large")

    @patch("milknado.app.plan._spawn_plan_critic")
    def test_reprompts_once_then_raises_on_persistent_unparseable_output(
        self, mock_spawn: MagicMock, tmp_path: Path
    ) -> None:
        mock_spawn.return_value = "garbage, no verdict tag"
        cfg = _cfg(tmp_path, plan_reviewer_agent="claude --model sonnet -p")

        with pytest.raises(ValueError, match="unparseable <verdict>"):
            run_plan_critic("goal", "manifest summary", cfg)

        assert mock_spawn.call_count == 2
        second_prompt = mock_spawn.call_args_list[1].args[1]
        assert "did not include a valid" in second_prompt


class TestSpawnPlanCritic:
    """The critic-spawn seam writes the prompt to a context file, forces stdout
    capture, spawns the reviewer in the project root, and returns its stdout."""

    def test_writes_context_and_returns_stdout(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from milknado.app.plan import _spawn_plan_critic

        captured: dict[str, object] = {}

        def fake_run(argv, **kwargs):  # noqa: ANN001, ANN003, ANN202
            captured["argv"] = argv
            captured["kwargs"] = kwargs
            return subprocess.CompletedProcess(argv, 0, stdout="<verdict>approve</verdict>")

        monkeypatch.setattr(subprocess, "run", fake_run)

        out = _spawn_plan_critic("claude --model sonnet -p", "REVIEW THIS PLAN", tmp_path)

        assert out == "<verdict>approve</verdict>"
        context = tmp_path / ".milknado" / "plan-critic-context.md"
        assert context.read_text(encoding="utf-8") == "REVIEW THIS PLAN"
        assert captured["kwargs"]["cwd"] == tmp_path
        assert captured["kwargs"]["stdout"] == subprocess.PIPE

    def test_returns_empty_string_when_agent_emits_no_stdout(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from milknado.app.plan import _spawn_plan_critic

        def fake_run(argv, **kwargs):  # noqa: ANN001, ANN003, ANN202
            return subprocess.CompletedProcess(argv, 0, stdout=None)

        monkeypatch.setattr(subprocess, "run", fake_run)

        assert _spawn_plan_critic("claude -p", "prompt", tmp_path) == ""


class TestRunPlanWithCritic:
    def test_returns_none_verdict_when_plan_reviewer_agent_unset(self, tmp_path: Path) -> None:
        planner = MagicMock()
        planner.launch.return_value = _plan_result()
        cfg = _cfg(tmp_path, plan_reviewer_agent=None)

        result, verdict = _run_plan_with_critic(
            planner, "goal", tmp_path, tmp_path / "spec.md", cfg
        )

        assert verdict is None
        assert result == _plan_result()
        planner.launch.assert_called_once()

    @patch("milknado.app.plan.run_plan_critic")
    def test_returns_early_on_approve(self, mock_critic: MagicMock, tmp_path: Path) -> None:
        planner = MagicMock()
        planner.launch.return_value = _plan_result()
        mock_critic.return_value = CriticVerdict(approved=True, feedback="")
        cfg = _cfg(
            tmp_path,
            plan_reviewer_agent="claude --model sonnet -p",
            plan_review_max_rounds=3,
        )

        result, verdict = _run_plan_with_critic(
            planner, "goal", tmp_path, tmp_path / "spec.md", cfg
        )

        assert verdict == CriticVerdict(approved=True, feedback="")
        assert result == _plan_result()
        planner.launch.assert_called_once()
        mock_critic.assert_called_once()

    @patch("milknado.app.plan.run_plan_critic")
    def test_loops_and_replans_on_revise(self, mock_critic: MagicMock, tmp_path: Path) -> None:
        first = _plan_result(change_count=1)
        second = _plan_result(change_count=2)
        planner = MagicMock()
        planner.launch.side_effect = [first, second]
        mock_critic.side_effect = [
            CriticVerdict(approved=False, feedback="add more detail"),
            CriticVerdict(approved=True, feedback=""),
        ]
        cfg = _cfg(
            tmp_path,
            plan_reviewer_agent="claude --model sonnet -p",
            plan_review_max_rounds=3,
        )

        result, verdict = _run_plan_with_critic(
            planner, "goal", tmp_path, tmp_path / "spec.md", cfg
        )

        assert result == second
        assert verdict == CriticVerdict(approved=True, feedback="")
        assert planner.launch.call_count == 2
        replan_goal = planner.launch.call_args_list[1].args[0]
        assert "add more detail" in replan_goal

    @patch("milknado.app.plan.run_plan_critic")
    def test_stops_at_round_cap_returning_last_unapproved_verdict(
        self, mock_critic: MagicMock, tmp_path: Path
    ) -> None:
        planner = MagicMock()
        planner.launch.return_value = _plan_result()
        mock_critic.return_value = CriticVerdict(approved=False, feedback="still not right")
        cfg = _cfg(
            tmp_path,
            plan_reviewer_agent="claude --model sonnet -p",
            plan_review_max_rounds=2,
        )

        result, verdict = _run_plan_with_critic(
            planner, "goal", tmp_path, tmp_path / "spec.md", cfg
        )

        assert verdict == CriticVerdict(approved=False, feedback="still not right")
        # initial launch + one replan per round == 1 + plan_review_max_rounds
        assert planner.launch.call_count == 1 + cfg.plan_review_max_rounds
        assert mock_critic.call_count == cfg.plan_review_max_rounds

    def test_repeated_noninteractive_policy_runs_are_equivalent(self, tmp_path: Path) -> None:
        planner = MagicMock()
        planner.launch.return_value = _plan_result()
        cfg = _cfg(tmp_path, plan_reviewer_agent=None)

        first = _run_plan_with_critic(planner, "goal", tmp_path, tmp_path / "spec.md", cfg)
        second = _run_plan_with_critic(planner, "goal", tmp_path, tmp_path / "spec.md", cfg)

        assert second == first
        assert planner.launch.call_count == 2


class TestExecPlanNonInteractive:
    @patch("milknado.app.plan.run_plan_critic")
    def test_raises_when_critic_never_approves_within_cap(
        self, mock_critic: MagicMock, tmp_path: Path
    ) -> None:
        planner = MagicMock()
        planner.launch.return_value = _plan_result()
        mock_critic.return_value = CriticVerdict(approved=False, feedback="nope")
        cfg = _cfg(
            tmp_path,
            plan_reviewer_agent="claude --model sonnet -p",
            plan_review_max_rounds=2,
        )

        with pytest.raises(typer.Exit) as exc_info:
            _exec_plan_non_interactive(planner, "goal", tmp_path, tmp_path / "spec.md", cfg)

        assert exc_info.value.exit_code == 1


class TestExecPlanInteractive:
    @patch("milknado.cli.plan._prompt_plan_action")
    @patch("milknado.app.plan.run_plan_critic")
    def test_unapproved_critic_prints_fallback_and_continues_to_human_gate(
        self,
        mock_critic: MagicMock,
        mock_prompt: MagicMock,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        from milknado.cli.plan import _exec_plan_interactive

        planner = MagicMock()
        planner.launch.return_value = _plan_result()
        mock_critic.return_value = CriticVerdict(approved=False, feedback="needs more detail")
        mock_prompt.return_value = "1"  # accept, so the loop returns without raising
        cfg = _cfg(
            tmp_path,
            plan_reviewer_agent="claude --model sonnet -p",
            plan_review_max_rounds=1,
        )

        _exec_plan_interactive(planner, "goal", tmp_path, tmp_path / "spec.md", 5, cfg)

        captured = capsys.readouterr()
        assert "falling back to manual review" in captured.out
        assert "needs more detail" in captured.out


def test_cli_plan_surfaces_critic_failures_in_both_modes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib

    cli_plan = importlib.import_module("milknado.cli.plan")

    cfg = default_config(tmp_path)
    monkeypatch.setattr(
        cli_plan,
        "_run_plan_with_critic",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(cli_plan.PlanCriticError("critic failed")),
    )
    with pytest.raises(typer.Exit) as non_interactive:
        cli_plan._exec_plan_non_interactive(None, "goal", tmp_path, tmp_path / "spec", cfg)
    assert non_interactive.value.exit_code == 1

    with pytest.raises(typer.Exit) as interactive:
        cli_plan._exec_plan_interactive(None, "goal", tmp_path, tmp_path / "spec", 1, cfg)
    assert interactive.value.exit_code == 1
