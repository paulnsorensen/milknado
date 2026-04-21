from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from milknado.cli import app
from milknado.domains.common import default_config
from milknado.domains.graph import MikadoGraph

runner = CliRunner()


def _seeded_graph(project_dir: Path) -> MikadoGraph:
    config = default_config(project_dir)
    config.db_path.parent.mkdir(parents=True, exist_ok=True)
    graph = MikadoGraph(config.db_path)
    graph.add_node("existing root")
    graph.close()
    return graph


def _write_spec(project_dir: Path, body: str = "# US-001 feature\nDo the thing.") -> Path:
    spec = project_dir / "spec.md"
    spec.write_text(body, encoding="utf-8")
    return spec


@pytest.fixture()
def project_dir(tmp_path: Path) -> Path:
    return tmp_path


def _minimal_manifest_stdout() -> str:
    import json

    payload = {
        "manifest_version": "milknado.plan.v2",
        "goal": "Test goal",
        "goal_summary": "A short test summary.",
        "changes": [
            {"id": "c1", "path": "src/foo.py", "description": "US-001 implement"},
            {"id": "c2", "path": "tests/test_foo.py", "description": "US-001 tests"},
        ],
    }
    return "```json\n" + json.dumps(payload) + "\n```"


def _gapped_manifest_stdout() -> str:
    """Manifest with US-001 impl only — no tests/ change → coverage gap."""
    import json

    payload = {
        "manifest_version": "milknado.plan.v2",
        "goal": "Test goal",
        "goal_summary": "A short test summary.",
        "changes": [
            {"id": "c1", "path": "src/foo.py", "description": "US-001 implement"},
        ],
    }
    return "```json\n" + json.dumps(payload) + "\n```"


def _patched_subprocess(side_effects: list[str] | None = None, stdout: str | None = None):
    """Context manager: patches TilthAdapter + subprocess.run for planner tests."""
    from milknado.domains.common.types import DegradationMarker

    tilth_mock = MagicMock()
    tilth_mock.return_value.structural_map.return_value = DegradationMarker(
        source="tilth", reason="mocked"
    )
    mock_run = MagicMock()
    if side_effects is not None:
        mock_run.side_effect = [MagicMock(returncode=0, stdout=s, stderr="") for s in side_effects]
    else:
        mock_run.return_value = MagicMock(
            returncode=0, stdout=stdout or _minimal_manifest_stdout(), stderr=""
        )
    return (
        patch("milknado.domains.planning.planner.TilthAdapter", tilth_mock),
        patch("milknado.domains.planning.planner.subprocess.run", mock_run),
        mock_run,
    )


@pytest.fixture()
def mock_planner_subprocess():
    from milknado.domains.common.types import DegradationMarker

    tilth_mock = MagicMock()
    tilth_mock.return_value.structural_map.return_value = DegradationMarker(
        source="tilth", reason="mocked"
    )
    with (
        patch("milknado.domains.planning.planner.TilthAdapter", tilth_mock),
        patch("milknado.domains.planning.planner.subprocess.run") as mock_run,
    ):
        mock_run.return_value = MagicMock(
            returncode=0, stdout=_minimal_manifest_stdout(), stderr=""
        )
        yield mock_run


@pytest.fixture()
def mock_crg_adapter():
    with patch("milknado.adapters.crg.CrgAdapter") as crg_cls:
        crg = crg_cls.return_value
        crg.ensure_graph.return_value = None
        crg.get_architecture_overview.return_value = {"communities": [], "entry_points": []}
        crg.list_communities.return_value = []
        crg.list_flows.return_value = []
        crg.get_bridge_nodes.return_value = []
        crg.get_hub_nodes.return_value = []
        crg.semantic_search_nodes.return_value = []
        yield crg_cls


class TestPlanMutualExclusion:
    def test_resume_and_reset_together_exit_1(
        self, project_dir: Path, mock_crg_adapter: MagicMock
    ) -> None:
        spec = _write_spec(project_dir)
        result = runner.invoke(
            app,
            [
                "plan",
                "--spec",
                str(spec),
                "--resume",
                "--reset",
                "--project-root",
                str(project_dir),
            ],
        )
        assert result.exit_code == 1
        assert "mutually exclusive" in result.output.lower()

    def test_resume_and_reset_together_no_subprocess(
        self, project_dir: Path, mock_crg_adapter: MagicMock
    ) -> None:
        spec = _write_spec(project_dir)
        with patch("milknado.domains.planning.planner.subprocess.run") as mock_run:
            runner.invoke(
                app,
                [
                    "plan",
                    "--spec",
                    str(spec),
                    "--resume",
                    "--reset",
                    "--project-root",
                    str(project_dir),
                ],
            )
            mock_run.assert_not_called()


class TestPlanFreshEmptyDB:
    def test_fresh_db_proceeds_without_flags(
        self,
        project_dir: Path,
        mock_crg_adapter: MagicMock,
        mock_planner_subprocess: MagicMock,
    ) -> None:
        spec = _write_spec(project_dir)
        result = runner.invoke(
            app,
            ["plan", "--spec", str(spec), "--project-root", str(project_dir)],
        )
        assert result.exit_code == 0
        mock_planner_subprocess.assert_called_once()

    def test_fresh_db_no_error_message(
        self,
        project_dir: Path,
        mock_crg_adapter: MagicMock,
        mock_planner_subprocess: MagicMock,
    ) -> None:
        spec = _write_spec(project_dir)
        result = runner.invoke(
            app,
            ["plan", "--spec", str(spec), "--project-root", str(project_dir)],
        )
        assert "existing plan" not in result.output


class TestPlanNonEmptyNoFlags:
    def test_non_empty_no_flags_exits_1(
        self, project_dir: Path, mock_crg_adapter: MagicMock
    ) -> None:
        _seeded_graph(project_dir)
        spec = _write_spec(project_dir)
        result = runner.invoke(
            app,
            ["plan", "--spec", str(spec), "--project-root", str(project_dir)],
        )
        assert result.exit_code == 1

    def test_non_empty_no_flags_error_message(
        self, project_dir: Path, mock_crg_adapter: MagicMock
    ) -> None:
        _seeded_graph(project_dir)
        spec = _write_spec(project_dir)
        result = runner.invoke(
            app,
            ["plan", "--spec", str(spec), "--project-root", str(project_dir)],
        )
        assert "existing plan with 1 nodes" in result.output
        assert "--resume" in result.output
        assert "--reset" in result.output

    def test_non_empty_no_flags_no_subprocess(
        self, project_dir: Path, mock_crg_adapter: MagicMock
    ) -> None:
        _seeded_graph(project_dir)
        spec = _write_spec(project_dir)
        with patch("milknado.domains.planning.planner.subprocess.run") as mock_run:
            runner.invoke(
                app,
                ["plan", "--spec", str(spec), "--project-root", str(project_dir)],
            )
            mock_run.assert_not_called()


class TestPlanResume:
    def test_resume_flag_appends_without_error(
        self,
        project_dir: Path,
        mock_crg_adapter: MagicMock,
        mock_planner_subprocess: MagicMock,
    ) -> None:
        _seeded_graph(project_dir)
        spec = _write_spec(project_dir)
        result = runner.invoke(
            app,
            [
                "plan",
                "--spec",
                str(spec),
                "--resume",
                "--project-root",
                str(project_dir),
            ],
        )
        assert result.exit_code == 0

    def test_resume_calls_subprocess(
        self,
        project_dir: Path,
        mock_crg_adapter: MagicMock,
        mock_planner_subprocess: MagicMock,
    ) -> None:
        _seeded_graph(project_dir)
        spec = _write_spec(project_dir)
        runner.invoke(
            app,
            [
                "plan",
                "--spec",
                str(spec),
                "--resume",
                "--project-root",
                str(project_dir),
            ],
        )
        mock_planner_subprocess.assert_called_once()


class TestPlanReset:
    def test_reset_drops_existing_nodes_before_plan(
        self,
        project_dir: Path,
        mock_crg_adapter: MagicMock,
        mock_planner_subprocess: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        import logging

        _seeded_graph(project_dir)
        spec = _write_spec(project_dir)
        with caplog.at_level(logging.INFO, logger="milknado"):
            runner.invoke(
                app,
                [
                    "plan",
                    "--spec",
                    str(spec),
                    "--reset",
                    "--project-root",
                    str(project_dir),
                ],
            )
        assert any("Dropped 1 nodes" in r.message for r in caplog.records)
        config = default_config(project_dir)
        graph = MikadoGraph(config.db_path)
        try:
            node_descriptions = [n.description for n in graph.get_all_nodes()]
        finally:
            graph.close()
        assert "existing root" not in node_descriptions

    def test_reset_proceeds_to_call_subprocess(
        self,
        project_dir: Path,
        mock_crg_adapter: MagicMock,
        mock_planner_subprocess: MagicMock,
    ) -> None:
        _seeded_graph(project_dir)
        spec = _write_spec(project_dir)
        result = runner.invoke(
            app,
            [
                "plan",
                "--spec",
                str(spec),
                "--reset",
                "--project-root",
                str(project_dir),
            ],
        )
        assert result.exit_code == 0
        mock_planner_subprocess.assert_called_once()


class TestPlanResetWithOrphanedWorktrees:
    def test_reset_warns_on_milknado_worktrees(
        self,
        project_dir: Path,
        mock_crg_adapter: MagicMock,
        mock_planner_subprocess: MagicMock,
    ) -> None:
        _seeded_graph(project_dir)
        (project_dir / ".worktrees" / "milknado-feature-abc").mkdir(parents=True)
        spec = _write_spec(project_dir)
        result = runner.invoke(
            app,
            [
                "plan",
                "--spec",
                str(spec),
                "--reset",
                "--project-root",
                str(project_dir),
            ],
        )
        assert "milknado-feature-abc" in result.stderr or "milknado-feature-abc" in result.output

    def test_reset_warns_on_claude_subdirectory_worktrees(
        self,
        project_dir: Path,
        mock_crg_adapter: MagicMock,
        mock_planner_subprocess: MagicMock,
    ) -> None:
        _seeded_graph(project_dir)
        (project_dir / ".worktrees" / "claude" / "milknado-task-123").mkdir(parents=True)
        spec = _write_spec(project_dir)
        result = runner.invoke(
            app,
            [
                "plan",
                "--spec",
                str(spec),
                "--reset",
                "--project-root",
                str(project_dir),
            ],
        )
        combined = result.stderr + result.output
        assert "milknado-task-123" in combined

    def test_reset_continues_after_warning(
        self,
        project_dir: Path,
        mock_crg_adapter: MagicMock,
        mock_planner_subprocess: MagicMock,
    ) -> None:
        _seeded_graph(project_dir)
        (project_dir / ".worktrees" / "milknado-orphan").mkdir(parents=True)
        spec = _write_spec(project_dir)
        result = runner.invoke(
            app,
            [
                "plan",
                "--spec",
                str(spec),
                "--reset",
                "--project-root",
                str(project_dir),
            ],
        )
        assert result.exit_code == 0
        mock_planner_subprocess.assert_called_once()


class TestPlanSpecHashMismatchWarning:
    def test_resume_warns_when_spec_hash_changed(
        self,
        project_dir: Path,
        mock_crg_adapter: MagicMock,
        mock_planner_subprocess: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        import logging

        _seeded_graph(project_dir)
        config = default_config(project_dir)
        graph = MikadoGraph(config.db_path)
        graph.set_spec_hash("aabbccdd" + "0" * 56)
        graph.close()

        spec = _write_spec(project_dir, "# US-001 totally different spec content")
        with caplog.at_level(logging.WARNING, logger="milknado"):
            runner.invoke(
                app,
                [
                    "plan",
                    "--spec",
                    str(spec),
                    "--resume",
                    "--project-root",
                    str(project_dir),
                ],
            )
        assert any("mismatch" in r.message for r in caplog.records)

    def test_resume_no_warn_when_hash_matches(
        self,
        project_dir: Path,
        mock_crg_adapter: MagicMock,
        mock_planner_subprocess: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        import logging

        spec_body = "# US-001 consistent spec content"
        _seeded_graph(project_dir)
        config = default_config(project_dir)
        graph = MikadoGraph(config.db_path)
        stored = hashlib.sha256(spec_body.encode()).hexdigest()
        graph.set_spec_hash(stored)
        graph.close()

        spec = _write_spec(project_dir, spec_body)
        with caplog.at_level(logging.WARNING, logger="milknado"):
            runner.invoke(
                app,
                [
                    "plan",
                    "--spec",
                    str(spec),
                    "--resume",
                    "--project-root",
                    str(project_dir),
                ],
            )
        assert not any("mismatch" in r.message for r in caplog.records)


class TestVerifyRoundCap:
    """US-003: max_verify_rounds flows through CLI and caps the coverage loop."""

    def test_cap_zero_subprocess_called_once(
        self,
        project_dir: Path,
        mock_crg_adapter: MagicMock,
    ) -> None:
        """cap=0 → coverage loop never runs; subprocess called exactly once."""
        spec = _write_spec(project_dir)
        tilth_patch, run_patch, mock_run = _patched_subprocess(stdout=_minimal_manifest_stdout())
        with tilth_patch, run_patch:
            result = runner.invoke(
                app,
                [
                    "plan",
                    "--spec",
                    str(spec),
                    "--max-verify-rounds",
                    "0",
                    "--project-root",
                    str(project_dir),
                ],
            )
        assert result.exit_code == 0
        assert mock_run.call_count == 1

    def test_cap_3_two_rounds_no_cap_hit(
        self,
        project_dir: Path,
        mock_crg_adapter: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """cap=3, gap fixed on 2nd call → loop exits cleanly, no cap-hit warning."""
        import logging

        spec = _write_spec(project_dir)
        tilth_patch, run_patch, mock_run = _patched_subprocess(
            side_effects=[_gapped_manifest_stdout(), _minimal_manifest_stdout()]
        )
        with tilth_patch, run_patch:
            with caplog.at_level(logging.WARNING, logger="milknado"):
                result = runner.invoke(
                    app,
                    [
                        "plan",
                        "--spec",
                        str(spec),
                        "--max-verify-rounds",
                        "3",
                        "--project-root",
                        str(project_dir),
                    ],
                )
        assert result.exit_code == 0
        assert mock_run.call_count == 2
        assert not any("round cap hit" in r.message for r in caplog.records)
        assert "CAP-HIT" not in result.output

    def test_cap_3_three_rounds_cap_hit_raises_insufficient_coverage(
        self,
        project_dir: Path,
        mock_crg_adapter: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """cap=3, all rounds return gaps → cap-hit warning logged, coverage error raised."""
        import logging

        from milknado.domains.common.errors import InsufficientTestCoverageError

        spec = _write_spec(project_dir)
        tilth_patch, run_patch, mock_run = _patched_subprocess(stdout=_gapped_manifest_stdout())
        with tilth_patch, run_patch:
            with caplog.at_level(logging.WARNING, logger="milknado"):
                result = runner.invoke(
                    app,
                    [
                        "plan",
                        "--spec",
                        str(spec),
                        "--max-verify-rounds",
                        "3",
                        "--project-root",
                        str(project_dir),
                    ],
                )
        assert result.exit_code != 0
        assert isinstance(result.exception, InsufficientTestCoverageError)
        assert any("round cap hit" in r.message for r in caplog.records)

    def test_max_verify_rounds_plumbed_from_cli_flag(
        self,
        project_dir: Path,
    ) -> None:
        """--max-verify-rounds N passes max_verify_rounds=N to planner.launch."""
        from milknado.domains.planning.planner import PlanResult

        spec = _write_spec(project_dir)
        with (
            patch("milknado.adapters.crg.CrgAdapter"),
            patch("milknado.domains.planning.Planner") as mock_planner_cls,
        ):
            mock_planner_cls.return_value.launch.return_value = PlanResult(
                success=True,
                exit_code=0,
                solver_status="OPTIMAL",
                change_count=2,
                batch_count=1,
                nodes_created=2,
                verify_rounds_used=1,
                verify_round_cap_hit=False,
                coverage_gaps_remaining=0,
            )
            runner.invoke(
                app,
                [
                    "plan",
                    "--spec",
                    str(spec),
                    "--max-verify-rounds",
                    "5",
                    "--project-root",
                    str(project_dir),
                ],
            )
        call_kwargs = mock_planner_cls.return_value.launch.call_args.kwargs
        assert call_kwargs.get("max_verify_rounds") == 5
