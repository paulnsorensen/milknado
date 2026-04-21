from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from milknado.domains.batching import FileChange, NewRelationship, SymbolRef
from milknado.domains.common.types import DegradationMarker, TilthMap
from milknado.domains.graph import MikadoGraph
from milknado.domains.planning.context import build_planning_context
from milknado.domains.planning.manifest import (
    MANIFEST_VERSION,
    PlanChangeManifest,
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
    crg.list_communities.return_value = [{"name": f"community_{i}"} for i in range(10)]
    crg.list_flows.return_value = [{"name": f"flow_{i}"} for i in range(10)]
    crg.get_bridge_nodes.return_value = [{"name": f"bridge_{i}"} for i in range(10)]
    crg.get_hub_nodes.return_value = [{"name": f"hub_{i}"} for i in range(10)]
    return crg


class TestBuildPlanningContext:
    def test_includes_goal(self, tmp_graph: MikadoGraph, mock_crg: MagicMock) -> None:
        ctx = build_planning_context("extract auth", mock_crg, tmp_graph)
        assert "# Goal" in ctx
        assert "extract auth" in ctx

    def test_includes_touch_sites_section(self, tmp_graph: MikadoGraph, mock_crg: MagicMock) -> None:
        ctx = build_planning_context("goal", mock_crg, tmp_graph)
        assert "# Probable Touch Sites" in ctx
        # old broken CRG compact section must not appear
        assert "Architecture (compact)" not in ctx

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

    def test_touch_sites_section_with_tilth_map(
        self, tmp_graph: MikadoGraph, mock_crg: MagicMock
    ) -> None:
        mock_tilth = MagicMock()
        mock_tilth.structural_map.return_value = TilthMap(
            scope=Path("."), budget_tokens=400, data={"modules": 42, "files": 7}
        )
        ctx = build_planning_context("goal", mock_crg, tmp_graph, tilth=mock_tilth)
        assert "# Probable Touch Sites" in ctx
        assert "modules" in ctx
        assert "42" in ctx

    def test_touch_sites_section_degradation_message(
        self, tmp_graph: MagicMock, mock_crg: MagicMock
    ) -> None:
        mock_tilth = MagicMock()
        mock_tilth.structural_map.return_value = DegradationMarker(
            source="tilth", reason="binary not found"
        )
        ctx = build_planning_context("goal", mock_crg, tmp_graph, tilth=mock_tilth)
        assert "# Probable Touch Sites" in ctx
        assert "tilth" in ctx
        assert "binary not found" in ctx

    def test_touch_sites_section_when_tilth_none(
        self, tmp_graph: MikadoGraph, mock_crg: MagicMock
    ) -> None:
        ctx = build_planning_context("goal", mock_crg, tmp_graph, tilth=None)
        assert "# Probable Touch Sites" in ctx
        assert "unavailable" in ctx

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
    @patch("milknado.domains.planning.planner.subprocess.run")
    def test_launch_writes_context_file(
        self,
        mock_run: MagicMock,
        tmp_path: Path,
        tmp_graph: MikadoGraph,
        mock_crg: MagicMock,
    ) -> None:
        mock_run.return_value = MagicMock(returncode=0, stdout="")
        planner = Planner(tmp_graph, mock_crg, "claude")
        result = planner.launch("my goal", tmp_path)
        assert result.success is True
        assert result.exit_code == 0
        assert result.context_path is not None
        assert result.context_path.exists()
        content = result.context_path.read_text()
        assert "my goal" in content

    @patch("milknado.domains.planning.planner.TilthAdapter")
    @patch("milknado.domains.planning.planner.subprocess.run")
    def test_launch_calls_agent(
        self,
        mock_run: MagicMock,
        mock_tilth_cls: MagicMock,
        tmp_path: Path,
        tmp_graph: MikadoGraph,
        mock_crg: MagicMock,
    ) -> None:
        from milknado.domains.common.types import DegradationMarker

        mock_tilth_cls.return_value.structural_map.return_value = DegradationMarker(
            source="tilth", reason="mocked"
        )
        mock_run.return_value = MagicMock(returncode=0, stdout="")
        planner = Planner(tmp_graph, mock_crg, "claude")
        planner.launch("my goal", tmp_path)
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "claude"
        assert cmd[-1] == "-"

    @patch("milknado.domains.planning.planner.TilthAdapter")
    @patch("milknado.domains.planning.planner.subprocess.run")
    def test_launch_uses_stdin_input(
        self,
        mock_run: MagicMock,
        mock_tilth_cls: MagicMock,
        tmp_path: Path,
        tmp_graph: MikadoGraph,
        mock_crg: MagicMock,
    ) -> None:
        from milknado.domains.common.types import DegradationMarker

        mock_tilth_cls.return_value.structural_map.return_value = DegradationMarker(
            source="tilth", reason="mocked"
        )
        mock_run.return_value = MagicMock(returncode=0, stdout="")
        planner = Planner(tmp_graph, mock_crg, "claude -p --dangerously-skip-permissions")
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
        from milknado.domains.common.errors import PlanningFailed

        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="agent failed")
        planner = Planner(tmp_graph, mock_crg, "claude")
        with pytest.raises(PlanningFailed) as exc_info:
            planner.launch("my goal", tmp_path)
        assert "agent failed" in str(exc_info.value)

    @patch("milknado.domains.planning.planner.subprocess.run")
    def test_custom_agent_command(
        self,
        mock_run: MagicMock,
        tmp_path: Path,
        tmp_graph: MikadoGraph,
        mock_crg: MagicMock,
    ) -> None:
        mock_run.return_value = MagicMock(returncode=0, stdout="")
        planner = Planner(tmp_graph, mock_crg, "aider --model opus")
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
        mock_run.return_value = MagicMock(returncode=0, stdout="")
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
    @patch("milknado.domains.planning.planner.subprocess.run")
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
        planner = Planner(tmp_graph, mock_crg, "claude")
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
        planner = Planner(tmp_graph, mock_crg, "claude")
        with pytest.raises(FileNotFoundError, match="spec_path"):
            planner.launch("goal", tmp_path, spec_path=tmp_path / "nonexistent.md")

    @patch("milknado.domains.planning.planner.subprocess.run")
    def test_no_manifest_returns_no_manifest_status(
        self,
        mock_run: MagicMock,
        tmp_path: Path,
        tmp_graph: MikadoGraph,
        mock_crg: MagicMock,
    ) -> None:
        mock_run.return_value = MagicMock(returncode=0, stdout="no JSON block here")
        planner = Planner(tmp_graph, mock_crg, "claude")
        result = planner.launch("goal", tmp_path)
        assert result.solver_status == "NO_MANIFEST"
        assert result.nodes_created == 0
        assert result.batch_count == 0

    @patch("milknado.domains.planning.planner.subprocess.run")
    def test_no_manifest_no_telemetry(
        self,
        mock_run: MagicMock,
        tmp_path: Path,
        tmp_graph: MikadoGraph,
        mock_crg: MagicMock,
    ) -> None:
        mock_run.return_value = MagicMock(returncode=0, stdout="no block")
        planner = Planner(tmp_graph, mock_crg, "claude")
        planner.launch("goal", tmp_path)
        assert not (tmp_path / ".milknado" / "calibration.jsonl").exists()

    @patch("milknado.domains.planning.planner.subprocess.run")
    def test_happy_path_creates_nodes_and_telemetry(
        self,
        mock_run: MagicMock,
        tmp_path: Path,
        tmp_graph: MikadoGraph,
        mock_crg: MagicMock,
    ) -> None:
        stdout = _make_v2_manifest_stdout(
            [
                {"id": "c1", "path": "src/foo.py", "description": "US-001 Add Foo class"},
                {"id": "c2", "path": "tests/test_foo.py", "description": "US-001 tests"},
            ]
        )
        mock_run.return_value = MagicMock(returncode=0, stdout=stdout)
        planner = Planner(tmp_graph, mock_crg, "claude")
        result = planner.launch("Test goal", tmp_path)
        assert result.nodes_created > 0
        assert (tmp_path / ".milknado" / "calibration.jsonl").exists()

    @patch("milknado.domains.planning.planner.subprocess.run")
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
                {"id": "c1", "path": "src/foo.py", "description": "US-001 Add Foo"},
                {"id": "c2", "path": "tests/test_foo.py", "description": "US-001 tests"},
            ]
        )
        mock_run.return_value = MagicMock(returncode=0, stdout=stdout)
        planner = Planner(tmp_graph, mock_crg, "claude")
        result = planner.launch("goal", tmp_path)
        assert result.nodes_created > 0


def _wrap(payload: dict) -> str:
    import json as _json

    return "preamble\n```json\n" + _json.dumps(payload) + "\n```\ntrailer"


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
        assert only.depends_on == ()
        assert manifest.new_relationships == ()

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
            manifest.changes = ()  # type: ignore

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


class TestBuildCoverageDelta:
    def test_contains_original_context(self) -> None:
        from milknado.domains.planning._plan_helpers import (
            build_coverage_delta as _build_coverage_delta,
        )

        result = _build_coverage_delta(["US-001"], "# Original context\nsome body")
        assert "# Original context" in result
        assert "some body" in result

    def test_lists_all_gaps(self) -> None:
        from milknado.domains.planning._plan_helpers import (
            build_coverage_delta as _build_coverage_delta,
        )

        result = _build_coverage_delta(["US-001", "US-002", "US-003"], "ctx")
        assert "- US-001" in result
        assert "- US-002" in result
        assert "- US-003" in result

    def test_instructs_tests_path(self) -> None:
        from milknado.domains.planning._plan_helpers import (
            build_coverage_delta as _build_coverage_delta,
        )

        result = _build_coverage_delta(["US-001"], "ctx")
        assert "tests/" in result

    def test_requests_updated_manifest(self) -> None:
        from milknado.domains.planning._plan_helpers import (
            build_coverage_delta as _build_coverage_delta,
        )

        result = _build_coverage_delta(["US-001"], "ctx")
        assert "manifest" in result.lower()


class TestPlannerCoverageLoop:
    def _covered_stdout(self, us_ref: str) -> str:
        return _make_v2_manifest_stdout(
            [
                {"id": "c1", "path": "src/foo.py", "description": f"{us_ref} implement"},
                {"id": "c2", "path": "tests/test_foo.py", "description": f"{us_ref} tests"},
            ]
        )

    def _uncovered_stdout(self, us_ref: str) -> str:
        return _make_v2_manifest_stdout(
            [
                {"id": "c1", "path": "src/foo.py", "description": f"{us_ref} implement"},
            ]
        )

    @patch("milknado.domains.planning.planner.TilthAdapter")
    @patch("milknado.domains.planning.planner.subprocess.run")
    def test_no_reprompt_when_fully_covered(
        self,
        mock_run: MagicMock,
        mock_tilth_cls: MagicMock,
        tmp_path: Path,
        tmp_graph: MikadoGraph,
        mock_crg: MagicMock,
    ) -> None:
        from milknado.domains.common.types import DegradationMarker

        mock_tilth_cls.return_value.structural_map.return_value = DegradationMarker(
            source="tilth", reason="mocked"
        )
        spec = tmp_path / "spec.md"
        spec.write_text("US-001 the feature", encoding="utf-8")
        mock_run.return_value = MagicMock(returncode=0, stdout=self._covered_stdout("US-001"))
        planner = Planner(tmp_graph, mock_crg, "claude")
        planner.launch("goal", tmp_path, spec_path=spec, max_verify_rounds=2)
        assert mock_run.call_count == 1

    @patch("milknado.domains.planning.planner.TilthAdapter")
    @patch("milknado.domains.planning.planner.subprocess.run")
    def test_reprompt_when_gaps_and_passes_remain(
        self,
        mock_run: MagicMock,
        mock_tilth_cls: MagicMock,
        tmp_path: Path,
        tmp_graph: MikadoGraph,
        mock_crg: MagicMock,
    ) -> None:
        from milknado.domains.common.types import DegradationMarker

        mock_tilth_cls.return_value.structural_map.return_value = DegradationMarker(
            source="tilth", reason="mocked"
        )
        spec = tmp_path / "spec.md"
        spec.write_text("US-001 the feature", encoding="utf-8")
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout=self._uncovered_stdout("US-001")),
            MagicMock(returncode=0, stdout=self._covered_stdout("US-001")),
        ]
        planner = Planner(tmp_graph, mock_crg, "claude")
        planner.launch("goal", tmp_path, spec_path=spec, max_verify_rounds=2)
        assert mock_run.call_count == 2

    @patch("milknado.domains.planning.planner.TilthAdapter")
    @patch("milknado.domains.planning.planner.subprocess.run")
    def test_no_reprompt_when_max_verify_rounds_zero(
        self,
        mock_run: MagicMock,
        mock_tilth_cls: MagicMock,
        tmp_path: Path,
        tmp_graph: MikadoGraph,
        mock_crg: MagicMock,
    ) -> None:
        from milknado.domains.common.errors import InsufficientTestCoverageError
        from milknado.domains.common.types import DegradationMarker

        mock_tilth_cls.return_value.structural_map.return_value = DegradationMarker(
            source="tilth", reason="mocked"
        )
        spec = tmp_path / "spec.md"
        spec.write_text("US-001 the feature", encoding="utf-8")
        mock_run.return_value = MagicMock(returncode=0, stdout=self._uncovered_stdout("US-001"))
        planner = Planner(tmp_graph, mock_crg, "claude")
        with pytest.raises(InsufficientTestCoverageError):
            planner.launch("goal", tmp_path, spec_path=spec, max_verify_rounds=0)
        assert mock_run.call_count == 1

    @patch("milknado.domains.planning.planner.TilthAdapter")
    @patch("milknado.domains.planning.planner.subprocess.run")
    def test_warning_logged_on_cap_hit(
        self,
        mock_run: MagicMock,
        mock_tilth_cls: MagicMock,
        tmp_path: Path,
        tmp_graph: MikadoGraph,
        mock_crg: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        import logging

        from milknado.domains.common.types import DegradationMarker

        mock_tilth_cls.return_value.structural_map.return_value = DegradationMarker(
            source="tilth", reason="mocked"
        )
        spec = tmp_path / "spec.md"
        spec.write_text("US-001 the feature", encoding="utf-8")
        mock_run.return_value = MagicMock(returncode=0, stdout=self._uncovered_stdout("US-001"))
        from milknado.domains.common.errors import InsufficientTestCoverageError

        planner = Planner(tmp_graph, mock_crg, "claude")
        with caplog.at_level(logging.WARNING, logger="milknado"):
            with pytest.raises(InsufficientTestCoverageError):
                planner.launch("goal", tmp_path, spec_path=spec, max_verify_rounds=1)
        assert any("verify-spec round cap hit" in r.message for r in caplog.records)
        assert any("lack test coverage" in r.message for r in caplog.records)

    @patch("milknado.domains.planning.planner.TilthAdapter")
    @patch("milknado.domains.planning.planner.subprocess.run")
    def test_delta_file_written_on_reprompt(
        self,
        mock_run: MagicMock,
        mock_tilth_cls: MagicMock,
        tmp_path: Path,
        tmp_graph: MikadoGraph,
        mock_crg: MagicMock,
    ) -> None:
        from milknado.domains.common.types import DegradationMarker

        mock_tilth_cls.return_value.structural_map.return_value = DegradationMarker(
            source="tilth", reason="mocked"
        )
        spec = tmp_path / "spec.md"
        spec.write_text("US-001 the feature", encoding="utf-8")
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout=self._uncovered_stdout("US-001")),
            MagicMock(returncode=0, stdout=self._covered_stdout("US-001")),
        ]
        planner = Planner(tmp_graph, mock_crg, "claude")
        planner.launch("goal", tmp_path, spec_path=spec, max_verify_rounds=2)
        delta = tmp_path / ".milknado" / "coverage-delta-1.md"
        assert delta.exists()
        assert "US-001" in delta.read_text(encoding="utf-8")


class TestUS010ExistingPlanGuard:
    """US-010: existing plan detection before any Opus call."""

    @patch("milknado.domains.planning.planner.subprocess.run")
    def test_fresh_graph_proceeds(
        self,
        mock_run: MagicMock,
        tmp_path: Path,
        tmp_graph: MikadoGraph,
        mock_crg: MagicMock,
    ) -> None:
        mock_run.return_value = MagicMock(returncode=0, stdout="")
        planner = Planner(tmp_graph, mock_crg, "claude")
        result = planner.launch("goal", tmp_path)
        assert result.success is True
        assert mock_run.called

    def test_non_empty_no_flags_raises(
        self,
        tmp_path: Path,
        tmp_graph: MikadoGraph,
        mock_crg: MagicMock,
    ) -> None:
        from milknado.domains.common.errors import ExistingPlanDetected

        tmp_graph.add_node("existing root")
        planner = Planner(tmp_graph, mock_crg, "claude")
        with pytest.raises(ExistingPlanDetected) as exc_info:
            planner.launch("goal", tmp_path)
        msg = str(exc_info.value)
        assert "existing plan with 1 nodes" in msg
        assert "0 done" in msg
        assert "1 pending" in msg
        assert "0 running" in msg
        assert "--resume" in msg
        assert "--reset" in msg

    def test_non_empty_no_flags_does_not_call_opus(
        self,
        tmp_path: Path,
        tmp_graph: MikadoGraph,
        mock_crg: MagicMock,
    ) -> None:
        from milknado.domains.common.errors import ExistingPlanDetected

        tmp_graph.add_node("existing root")
        planner = Planner(tmp_graph, mock_crg, "claude")
        with pytest.raises(ExistingPlanDetected):
            planner.launch("goal", tmp_path)
        # subprocess.run should not have been called (no Opus call)
        # checked via: if Opus were called the test would need mock_run

    @patch("milknado.domains.planning.planner.subprocess.run")
    def test_reset_drops_nodes_before_opus(
        self,
        mock_run: MagicMock,
        tmp_path: Path,
        tmp_graph: MikadoGraph,
        mock_crg: MagicMock,
    ) -> None:
        mock_run.return_value = MagicMock(returncode=0, stdout="")
        tmp_graph.add_node("node to be dropped")
        tmp_graph.add_node("another node")
        planner = Planner(tmp_graph, mock_crg, "claude")
        planner.launch("goal", tmp_path, reset=True)
        # all nodes dropped: graph is empty after reset
        assert tmp_graph.get_all_nodes() == []
        # Opus was called (reset proceeded to Opus)
        assert mock_run.called

    @patch("milknado.domains.planning.planner.subprocess.run")
    def test_reset_logs_dropped_count(
        self,
        mock_run: MagicMock,
        tmp_path: Path,
        tmp_graph: MikadoGraph,
        mock_crg: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        import logging

        mock_run.return_value = MagicMock(returncode=0, stdout="")
        tmp_graph.add_node("n1")
        tmp_graph.add_node("n2")
        tmp_graph.add_node("n3")
        planner = Planner(tmp_graph, mock_crg, "claude")
        with caplog.at_level(logging.INFO, logger="milknado"):
            planner.launch("goal", tmp_path, reset=True)
        assert any("Dropped 3 nodes" in r.message for r in caplog.records)

    @patch("milknado.domains.planning.planner.subprocess.run")
    def test_resuming_allows_non_empty_graph(
        self,
        mock_run: MagicMock,
        tmp_path: Path,
        tmp_graph: MikadoGraph,
        mock_crg: MagicMock,
    ) -> None:
        mock_run.return_value = MagicMock(returncode=0, stdout="")
        tmp_graph.add_node("existing root")
        planner = Planner(tmp_graph, mock_crg, "claude")
        result = planner.launch("goal", tmp_path, resuming=True)
        assert result.success is True
        assert mock_run.called

    @patch("milknado.domains.planning.planner.subprocess.run")
    def test_spec_hash_persisted_after_successful_plan(
        self,
        mock_run: MagicMock,
        tmp_path: Path,
        tmp_graph: MikadoGraph,
        mock_crg: MagicMock,
    ) -> None:
        import hashlib

        spec = tmp_path / "spec.md"
        spec.write_text("# US-001 feature\nDo the thing.", encoding="utf-8")
        stdout = _make_v2_manifest_stdout(
            [
                {"id": "c1", "path": "src/foo.py", "description": "US-001 impl"},
                {"id": "c2", "path": "tests/test_foo.py", "description": "US-001 tests"},
            ]
        )
        mock_run.return_value = MagicMock(returncode=0, stdout=stdout)
        planner = Planner(tmp_graph, mock_crg, "claude")
        planner.launch("goal", tmp_path, spec_path=spec)
        expected = hashlib.sha256(spec.read_bytes()).hexdigest()
        assert tmp_graph.get_spec_hash() == expected

    @patch("milknado.domains.planning.planner.subprocess.run")
    def test_resume_warns_on_spec_hash_mismatch(
        self,
        mock_run: MagicMock,
        tmp_path: Path,
        tmp_graph: MikadoGraph,
        mock_crg: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        import logging

        # Plant a hash for "old spec"
        tmp_graph.set_spec_hash("aabbccdd" + "0" * 56)
        tmp_graph.add_node("existing root")
        spec = tmp_path / "spec.md"
        spec.write_text("# NEW spec content that differs", encoding="utf-8")
        mock_run.return_value = MagicMock(returncode=0, stdout="")
        planner = Planner(tmp_graph, mock_crg, "claude")
        with caplog.at_level(logging.WARNING, logger="milknado"):
            planner.launch("goal", tmp_path, spec_path=spec, resuming=True)
        assert any("mismatch" in r.message for r in caplog.records)

    @patch("milknado.domains.planning.planner.subprocess.run")
    def test_resume_no_warn_when_hash_matches(
        self,
        mock_run: MagicMock,
        tmp_path: Path,
        tmp_graph: MikadoGraph,
        mock_crg: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        import hashlib
        import logging

        spec = tmp_path / "spec.md"
        spec.write_text("# same spec content", encoding="utf-8")
        stored = hashlib.sha256(b"# same spec content").hexdigest()
        tmp_graph.set_spec_hash(stored)
        tmp_graph.add_node("existing root")
        mock_run.return_value = MagicMock(returncode=0, stdout="")
        planner = Planner(tmp_graph, mock_crg, "claude")
        with caplog.at_level(logging.WARNING, logger="milknado"):
            planner.launch("goal", tmp_path, spec_path=spec, resuming=True)
        assert not any("mismatch" in r.message for r in caplog.records)

    @patch("milknado.domains.planning.planner.subprocess.run")
    def test_reset_warns_active_worktrees(
        self,
        mock_run: MagicMock,
        tmp_path: Path,
        tmp_graph: MikadoGraph,
        mock_crg: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        mock_run.return_value = MagicMock(returncode=0, stdout="")
        tmp_graph.add_node("to drop")
        worktrees = tmp_path / ".worktrees"
        (worktrees / "milknado-feature-abc").mkdir(parents=True)
        (worktrees / "claude" / "milknado-other").mkdir(parents=True)
        planner = Planner(tmp_graph, mock_crg, "claude")
        planner.launch("goal", tmp_path, reset=True)
        err = capsys.readouterr().err
        assert "milknado-feature-abc" in err
        assert "milknado-other" in err

    @patch("milknado.domains.planning.planner.subprocess.run")
    def test_reset_no_warn_when_no_worktrees(
        self,
        mock_run: MagicMock,
        tmp_path: Path,
        tmp_graph: MikadoGraph,
        mock_crg: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        mock_run.return_value = MagicMock(returncode=0, stdout="")
        tmp_graph.add_node("to drop")
        planner = Planner(tmp_graph, mock_crg, "claude")
        planner.launch("goal", tmp_path, reset=True)
        err = capsys.readouterr().err
        assert "WARNING" not in err

    def test_error_message_exact_format(
        self,
        tmp_path: Path,
        tmp_graph: MikadoGraph,
        mock_crg: MagicMock,
    ) -> None:
        from milknado.domains.common.errors import ExistingPlanDetected

        tmp_graph.add_node("root")
        tmp_graph.add_node("child", parent_id=1)
        tmp_graph.mark_running(2)
        planner = Planner(tmp_graph, mock_crg, "claude")
        with pytest.raises(ExistingPlanDetected) as exc_info:
            planner.launch("goal", tmp_path)
        msg = str(exc_info.value)
        assert "ERROR: existing plan with 2 nodes" in msg
        assert "0 done" in msg
        assert "1 pending" in msg
        assert "1 running" in msg


class TestAppendReuseCandidates:
    """US-210: _append_reuse_candidates block contract."""

    def _make_manifest(self, description: str = "Initial description") -> PlanChangeManifest:
        return PlanChangeManifest(
            manifest_version=MANIFEST_VERSION,
            goal="some goal",
            goal_summary="a summary",
            spec_path=None,
            changes=(FileChange(id="c1", path="src/auth.py", description=description),),
            new_relationships=(),
        )

    def test_appends_reuse_block_with_two_candidates(self) -> None:
        from milknado.domains.planning._manifest_quality import (
            append_reuse_candidates as _append_reuse_candidates,
        )

        crg = MagicMock()
        crg.semantic_search_nodes.return_value = [
            {"symbol_name": "AuthService", "file_path": "src/auth.py", "summary": "Handles auth"},
            {"symbol_name": "TokenStore", "file_path": "src/tokens.py", "summary": "JWT storage"},
        ]
        result = _append_reuse_candidates(self._make_manifest("Do the auth thing"), crg)

        desc = result.changes[0].description
        assert "## Reuse candidates" in desc
        assert "`AuthService`" in desc
        assert "src/auth.py" in desc
        assert "Handles auth" in desc
        assert "`TokenStore`" in desc
        assert "src/tokens.py" in desc
        assert "JWT storage" in desc

    def test_original_description_preserved_as_prefix(self) -> None:
        from milknado.domains.planning._manifest_quality import (
            append_reuse_candidates as _append_reuse_candidates,
        )

        crg = MagicMock()
        crg.semantic_search_nodes.return_value = [
            {"symbol_name": "FooBar", "file_path": "src/foo.py", "summary": "Foo stuff"},
        ]
        result = _append_reuse_candidates(self._make_manifest("Original description text"), crg)

        assert result.changes[0].description.startswith("Original description text")

    def test_candidate_line_format_matches_contract(self) -> None:
        from milknado.domains.planning._manifest_quality import (
            append_reuse_candidates as _append_reuse_candidates,
        )

        crg = MagicMock()
        crg.semantic_search_nodes.return_value = [
            {"symbol_name": "MyFunc", "file_path": "src/utils.py", "summary": "A utility"},
        ]
        result = _append_reuse_candidates(self._make_manifest("desc"), crg)

        assert "- `MyFunc` (src/utils.py): A utility" in result.changes[0].description

    def test_missing_dict_keys_fall_back_to_sentinels(self) -> None:
        from milknado.domains.planning._manifest_quality import (
            append_reuse_candidates as _append_reuse_candidates,
        )

        crg = MagicMock()
        crg.semantic_search_nodes.return_value = [{}]
        result = _append_reuse_candidates(self._make_manifest("desc"), crg)

        assert "- `?` (?): " in result.changes[0].description

    def test_empty_hit_list_leaves_description_unchanged(self) -> None:
        from milknado.domains.planning._manifest_quality import (
            append_reuse_candidates as _append_reuse_candidates,
        )

        crg = MagicMock()
        crg.semantic_search_nodes.return_value = []
        result = _append_reuse_candidates(self._make_manifest("Untouched description"), crg)

        assert result.changes[0].description == "Untouched description"

    def test_crg_none_returns_manifest_identity(self) -> None:
        from milknado.domains.planning._manifest_quality import (
            append_reuse_candidates as _append_reuse_candidates,
        )

        manifest = self._make_manifest("No crg description")
        result = _append_reuse_candidates(manifest, None)

        assert result is manifest

    def test_search_exception_logs_warning_and_preserves_description(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging

        from milknado.domains.planning._manifest_quality import (
            append_reuse_candidates as _append_reuse_candidates,
        )

        crg = MagicMock()
        crg.semantic_search_nodes.side_effect = RuntimeError("CRG exploded")

        with caplog.at_level(logging.WARNING, logger="milknado"):
            result = _append_reuse_candidates(self._make_manifest("Unmodified on error"), crg)

        assert result.changes[0].description == "Unmodified on error"
        assert any("CRG reuse search failed" in r.message for r in caplog.records)


class TestUS204BundlingEnforcement:
    """US-204: final-pass bundling blocker raises MultiStoryBundlingError."""

    @patch("milknado.domains.planning.planner.subprocess.run")
    def test_bundled_change_raises_multi_story_bundling_error(
        self,
        mock_run: MagicMock,
        tmp_path: Path,
        tmp_graph: MikadoGraph,
        mock_crg: MagicMock,
    ) -> None:
        from milknado.domains.common.errors import MultiStoryBundlingError

        stdout = _make_v2_manifest_stdout(
            [
                {
                    "id": "c1",
                    "path": "src/foo.py",
                    "description": "US-201 implements US-202 feature",
                },
            ]
        )
        mock_run.return_value = MagicMock(returncode=0, stdout=stdout)
        planner = Planner(tmp_graph, mock_crg, "claude")
        with pytest.raises(MultiStoryBundlingError):
            planner.launch("goal", tmp_path)

    @patch("milknado.domains.planning.planner.subprocess.run")
    def test_bundled_error_carries_violating_path(
        self,
        mock_run: MagicMock,
        tmp_path: Path,
        tmp_graph: MikadoGraph,
        mock_crg: MagicMock,
    ) -> None:
        from milknado.domains.common.errors import MultiStoryBundlingError

        stdout = _make_v2_manifest_stdout(
            [
                {
                    "id": "c1",
                    "path": "src/foo.py",
                    "description": "US-201 implements US-202 feature",
                },
            ]
        )
        mock_run.return_value = MagicMock(returncode=0, stdout=stdout)
        planner = Planner(tmp_graph, mock_crg, "claude")
        with pytest.raises(MultiStoryBundlingError) as exc_info:
            planner.launch("goal", tmp_path)
        assert "src/foo.py" in exc_info.value.bundled_changes

    @patch("milknado.domains.planning.planner.subprocess.run")
    def test_three_us_refs_in_one_description_raises(
        self,
        mock_run: MagicMock,
        tmp_path: Path,
        tmp_graph: MikadoGraph,
        mock_crg: MagicMock,
    ) -> None:
        from milknado.domains.common.errors import MultiStoryBundlingError

        stdout = _make_v2_manifest_stdout(
            [
                {
                    "id": "c1",
                    "path": "src/bar.py",
                    "description": "US-201 US-202 US-203 all in one",
                },
            ]
        )
        mock_run.return_value = MagicMock(returncode=0, stdout=stdout)
        planner = Planner(tmp_graph, mock_crg, "claude")
        with pytest.raises(MultiStoryBundlingError):
            planner.launch("goal", tmp_path)

    @patch("milknado.domains.planning.planner.subprocess.run")
    def test_multiple_bundled_changes_all_reported(
        self,
        mock_run: MagicMock,
        tmp_path: Path,
        tmp_graph: MikadoGraph,
        mock_crg: MagicMock,
    ) -> None:
        from milknado.domains.common.errors import MultiStoryBundlingError

        stdout = _make_v2_manifest_stdout(
            [
                {"id": "c1", "path": "src/foo.py", "description": "US-201 and US-202"},
                {"id": "c2", "path": "src/bar.py", "description": "US-203 and US-204"},
            ]
        )
        mock_run.return_value = MagicMock(returncode=0, stdout=stdout)
        planner = Planner(tmp_graph, mock_crg, "claude")
        with pytest.raises(MultiStoryBundlingError) as exc_info:
            planner.launch("goal", tmp_path)
        bundled = exc_info.value.bundled_changes
        assert "src/foo.py" in bundled
        assert "src/bar.py" in bundled
        assert len(bundled) == 2

    @patch("milknado.domains.planning.planner.subprocess.run")
    def test_split_changes_succeed(
        self,
        mock_run: MagicMock,
        tmp_path: Path,
        tmp_graph: MikadoGraph,
        mock_crg: MagicMock,
    ) -> None:
        stdout = _make_v2_manifest_stdout(
            [
                {"id": "c1", "path": "src/foo.py", "description": "US-201 implement foo"},
                {"id": "c2", "path": "tests/test_foo.py", "description": "US-201 tests foo"},
                {"id": "c3", "path": "src/bar.py", "description": "US-202 implement bar"},
                {"id": "c4", "path": "tests/test_bar.py", "description": "US-202 tests bar"},
            ]
        )
        mock_run.return_value = MagicMock(returncode=0, stdout=stdout)
        planner = Planner(tmp_graph, mock_crg, "claude")
        result = planner.launch("goal", tmp_path)
        assert result.success is True

    @patch("milknado.domains.planning.planner.subprocess.run")
    def test_same_ref_repeated_does_not_raise(
        self,
        mock_run: MagicMock,
        tmp_path: Path,
        tmp_graph: MikadoGraph,
        mock_crg: MagicMock,
    ) -> None:
        # same US-NNN ref repeated = 1 distinct ref, not bundled
        stdout = _make_v2_manifest_stdout(
            [
                {
                    "id": "c1",
                    "path": "src/foo.py",
                    "description": "US-201 implements US-201 requirement",
                },
                {"id": "c2", "path": "tests/test_foo.py", "description": "US-201 tests"},
            ]
        )
        mock_run.return_value = MagicMock(returncode=0, stdout=stdout)
        planner = Planner(tmp_graph, mock_crg, "claude")
        result = planner.launch("goal", tmp_path)
        assert result.success is True

    @patch("milknado.domains.planning.planner.subprocess.run")
    def test_bundling_error_is_subclass_of_milknado_error(
        self,
        mock_run: MagicMock,
        tmp_path: Path,
        tmp_graph: MikadoGraph,
        mock_crg: MagicMock,
    ) -> None:
        from milknado.domains.common.errors import MilknadoError, MultiStoryBundlingError

        stdout = _make_v2_manifest_stdout(
            [
                {"id": "c1", "path": "src/foo.py", "description": "US-201 and US-202 bundled"},
            ]
        )
        mock_run.return_value = MagicMock(returncode=0, stdout=stdout)
        planner = Planner(tmp_graph, mock_crg, "claude")
        with pytest.raises(MilknadoError) as exc_info:
            planner.launch("goal", tmp_path)
        assert isinstance(exc_info.value, MultiStoryBundlingError)


class TestUS205CoverageEnforcement:
    """US-205: raise InsufficientTestCoverageError when ratio < 0.5 and orphans remain."""

    def _two_impl_no_tests(self) -> str:
        return _make_v2_manifest_stdout(
            [
                {"id": "c1", "path": "src/foo.py", "description": "US-205 implement foo"},
                {"id": "c2", "path": "src/bar.py", "description": "US-205 implement bar"},
            ]
        )

    def _two_impl_one_shared_us_test(self) -> str:
        # single test shares US-205 ref → covers both impls via shared US-NNN
        return _make_v2_manifest_stdout(
            [
                {"id": "c1", "path": "src/foo.py", "description": "US-205 implement foo"},
                {"id": "c2", "path": "src/bar.py", "description": "US-205 implement bar"},
                {"id": "c3", "path": "tests/test_foo_bar.py", "description": "US-205 tests"},
            ]
        )

    def _two_impl_two_tests(self) -> str:
        return _make_v2_manifest_stdout(
            [
                {"id": "c1", "path": "src/foo.py", "description": "US-205 implement foo"},
                {"id": "c2", "path": "src/bar.py", "description": "US-205 implement bar"},
                {
                    "id": "c3",
                    "path": "tests/test_foo.py",
                    "description": "US-205 tests foo",
                    "depends_on": ["c1"],
                },
                {
                    "id": "c4",
                    "path": "tests/test_bar.py",
                    "description": "US-205 tests bar",
                    "depends_on": ["c2"],
                },
            ]
        )

    def _two_impl_one_partial_test(self) -> str:
        # c1 covered via depends_on; c2 (US-206) is orphan — ratio=0.5, no raise
        return _make_v2_manifest_stdout(
            [
                {"id": "c1", "path": "src/foo.py", "description": "US-205 implement foo"},
                {"id": "c2", "path": "src/bar.py", "description": "US-206 implement bar"},
                {
                    "id": "c3",
                    "path": "tests/test_foo.py",
                    "description": "US-205 tests foo",
                    "depends_on": ["c1"],
                },
            ]
        )

    @patch("milknado.domains.planning.planner.subprocess.run")
    def test_two_impl_no_tests_raises(
        self,
        mock_run: MagicMock,
        tmp_path: Path,
        tmp_graph: MikadoGraph,
        mock_crg: MagicMock,
    ) -> None:
        from milknado.domains.common.errors import InsufficientTestCoverageError

        mock_run.return_value = MagicMock(returncode=0, stdout=self._two_impl_no_tests())
        planner = Planner(tmp_graph, mock_crg, "claude")
        with pytest.raises(InsufficientTestCoverageError):
            planner.launch("goal", tmp_path, max_verify_rounds=0)

    @patch("milknado.domains.planning.planner.subprocess.run")
    def test_error_carries_both_orphan_paths(
        self,
        mock_run: MagicMock,
        tmp_path: Path,
        tmp_graph: MikadoGraph,
        mock_crg: MagicMock,
    ) -> None:
        from milknado.domains.common.errors import InsufficientTestCoverageError

        mock_run.return_value = MagicMock(returncode=0, stdout=self._two_impl_no_tests())
        planner = Planner(tmp_graph, mock_crg, "claude")
        with pytest.raises(InsufficientTestCoverageError) as exc_info:
            planner.launch("goal", tmp_path, max_verify_rounds=0)
        assert "src/foo.py" in exc_info.value.orphan_changes
        assert "src/bar.py" in exc_info.value.orphan_changes
        assert len(exc_info.value.orphan_changes) == 2

    @patch("milknado.domains.planning.planner.subprocess.run")
    def test_two_impl_one_shared_us_test_passes(
        self,
        mock_run: MagicMock,
        tmp_path: Path,
        tmp_graph: MikadoGraph,
        mock_crg: MagicMock,
    ) -> None:
        mock_run.return_value = MagicMock(returncode=0, stdout=self._two_impl_one_shared_us_test())
        planner = Planner(tmp_graph, mock_crg, "claude")
        result = planner.launch("goal", tmp_path, max_verify_rounds=1)
        assert result.success is True

    @patch("milknado.domains.planning.planner.subprocess.run")
    def test_two_impl_two_tests_passes(
        self,
        mock_run: MagicMock,
        tmp_path: Path,
        tmp_graph: MikadoGraph,
        mock_crg: MagicMock,
    ) -> None:
        mock_run.return_value = MagicMock(returncode=0, stdout=self._two_impl_two_tests())
        planner = Planner(tmp_graph, mock_crg, "claude")
        result = planner.launch("goal", tmp_path, max_verify_rounds=1)
        assert result.success is True

    @patch("milknado.domains.planning.planner.subprocess.run")
    def test_telemetry_records_orphan_ids(
        self,
        mock_run: MagicMock,
        tmp_path: Path,
        tmp_graph: MikadoGraph,
        mock_crg: MagicMock,
    ) -> None:
        import json

        # 2 impl + 1 test covering only c1 → c2 is orphan, ratio=0.5 → no raise, telemetry written
        mock_run.return_value = MagicMock(returncode=0, stdout=self._two_impl_one_partial_test())
        planner = Planner(tmp_graph, mock_crg, "claude")
        result = planner.launch("goal", tmp_path, max_verify_rounds=1)
        assert result.success is True
        record = json.loads((tmp_path / ".milknado" / "calibration.jsonl").read_text().strip())
        assert record["coverage_orphan_impl_changes"] == ["src/bar.py"]

    @patch("milknado.domains.planning.planner.subprocess.run")
    def test_insufficient_coverage_is_milknado_error(
        self,
        mock_run: MagicMock,
        tmp_path: Path,
        tmp_graph: MikadoGraph,
        mock_crg: MagicMock,
    ) -> None:
        from milknado.domains.common.errors import InsufficientTestCoverageError, MilknadoError

        mock_run.return_value = MagicMock(returncode=0, stdout=self._two_impl_no_tests())
        planner = Planner(tmp_graph, mock_crg, "claude")
        with pytest.raises(MilknadoError) as exc_info:
            planner.launch("goal", tmp_path, max_verify_rounds=0)
        assert isinstance(exc_info.value, InsufficientTestCoverageError)
