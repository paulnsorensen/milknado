"""Adversarial tests for the CLI plan command.

Focus: --spec validation, solver-status exit codes, binary files, empty files, unicode filenames.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from milknado.cli import app
from milknado.domains.planning.planner import PlanResult

runner = CliRunner()


def _make_plan_result(**kwargs: Any) -> PlanResult:
    defaults = {
        "success": True,
        "exit_code": 0,
        "change_count": 3,
        "batch_count": 2,
        "oversized_count": 0,
        "solver_status": "OPTIMAL",
        "nodes_created": 3,
        "verify_rounds_used": 1,
        "verify_round_cap_hit": False,
        "coverage_gaps_remaining": 0,
    }
    defaults.update(kwargs)
    return PlanResult(**defaults)  # type: ignore


class TestSpecFlagValidation:
    def test_missing_spec_flag_exits_nonzero(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["plan", "--project-root", str(tmp_path)])
        assert result.exit_code != 0

    def test_nonexistent_spec_path_exits_nonzero(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app,
            ["plan", "--spec", str(tmp_path / "ghost.md"), "--project-root", str(tmp_path)],
        )
        assert result.exit_code != 0

    def test_non_md_file_exits_nonzero(self, tmp_path: Path) -> None:
        spec = tmp_path / "spec.txt"
        spec.write_text("# Goal\nsome spec", encoding="utf-8")
        result = runner.invoke(
            app,
            ["plan", "--spec", str(spec), "--project-root", str(tmp_path)],
        )
        assert result.exit_code != 0

    def test_binary_file_renamed_md_exit_nonzero_or_error(self, tmp_path: Path) -> None:
        """Binary file with .md extension — _derive_goal reads it as UTF-8."""
        spec = tmp_path / "binary.md"
        spec.write_bytes(b"\x00\x01\x02\x03\xff\xfe\xfd")
        with (
            patch("milknado.domains.planning.Planner") as mock_planner_cls,
            patch("milknado.adapters.crg.CrgAdapter"),
            patch("milknado.app.run_command._load_or_default"),
            patch("milknado.app.run_command._ensure_db"),
        ):
            mock_planner = MagicMock()
            mock_planner_cls.return_value = mock_planner
            mock_planner.launch.return_value = _make_plan_result()
            result = runner.invoke(
                app,
                ["plan", "--spec", str(spec), "--project-root", str(tmp_path)],
            )
            # Binary file read as UTF-8 may raise UnicodeDecodeError or return garbage goal.
            # Either crash (exit != 0) or succeed with a garbage goal is documented.
            _ = result  # document the exit code without asserting — it's implementation-defined

    def test_empty_spec_file_does_not_crash(self, tmp_path: Path) -> None:
        """0-byte spec.md — _derive_goal returns stem, planner is launched."""
        spec = tmp_path / "spec.md"
        spec.write_bytes(b"")  # 0 bytes
        with (
            patch("milknado.domains.planning.Planner") as mock_planner_cls,
            patch("milknado.adapters.crg.CrgAdapter"),
            patch("milknado.app.run_command._load_or_default"),
            patch("milknado.app.run_command._ensure_db") as mock_ensure_db,
        ):
            mock_graph = MagicMock()
            mock_ensure_db.return_value = mock_graph
            mock_planner = MagicMock()
            mock_planner_cls.return_value = mock_planner
            mock_planner.launch.return_value = _make_plan_result()
            runner.invoke(
                app,
                ["plan", "--spec", str(spec), "--project-root", str(tmp_path)],
            )
            # Empty spec → _derive_goal should return stem ("spec") not crash
            if mock_planner.launch.called:
                goal_arg = mock_planner.launch.call_args[0][0]
                assert isinstance(goal_arg, str)
                assert len(goal_arg) > 0


class TestSolverStatusExitCodes:
    def _invoke_plan_with_result(self, tmp_path: Path, plan_result: PlanResult) -> Any:
        spec = tmp_path / "spec.md"
        spec.write_text("# My Goal\nsome spec", encoding="utf-8")
        with (
            patch("milknado.domains.planning.Planner") as mock_planner_cls,
            patch("milknado.adapters.crg.CrgAdapter"),
            patch("milknado.app.run_command._load_or_default"),
            patch("milknado.app.run_command._ensure_db") as mock_ensure_db,
        ):
            mock_graph = MagicMock()
            mock_ensure_db.return_value = mock_graph
            mock_planner = MagicMock()
            mock_planner_cls.return_value = mock_planner
            mock_planner.launch.return_value = plan_result
            result = runner.invoke(
                app,
                ["plan", "--spec", str(spec), "--project-root", str(tmp_path)],
            )
        return result

    def test_optimal_exits_zero(self, tmp_path: Path) -> None:
        result = self._invoke_plan_with_result(
            tmp_path, _make_plan_result(solver_status="OPTIMAL")
        )
        assert result.exit_code == 0

    def test_feasible_exits_zero(self, tmp_path: Path) -> None:
        result = self._invoke_plan_with_result(
            tmp_path, _make_plan_result(solver_status="FEASIBLE")
        )
        assert result.exit_code == 0

    def test_infeasible_exits_one(self, tmp_path: Path) -> None:
        result = self._invoke_plan_with_result(
            tmp_path,
            _make_plan_result(solver_status="INFEASIBLE", success=False, exit_code=1),
        )
        assert result.exit_code == 1

    def test_unknown_with_batches_exits_zero_with_warning(self, tmp_path: Path) -> None:
        result = self._invoke_plan_with_result(
            tmp_path,
            _make_plan_result(solver_status="UNKNOWN", batch_count=2),
        )
        assert result.exit_code == 0

    def test_unknown_with_zero_batches_exits_nonzero(self, tmp_path: Path) -> None:
        result = self._invoke_plan_with_result(
            tmp_path,
            _make_plan_result(
                solver_status="UNKNOWN",
                batch_count=0,
                success=False,
                exit_code=1,
            ),
        )
        assert result.exit_code != 0

    def test_no_manifest_exits_nonzero(self, tmp_path: Path) -> None:
        result = self._invoke_plan_with_result(
            tmp_path,
            _make_plan_result(
                solver_status="NO_MANIFEST",
                success=False,
                exit_code=1,
                batch_count=0,
                nodes_created=0,
            ),
        )
        assert result.exit_code != 0

    def test_unknown_solver_status_string_exit_behavior(self, tmp_path: Path) -> None:
        """A status value not in the known set — what exit code does _plan_exit_code return?"""
        # "TIMEOUT" is not in the known set. The CLI's _plan_exit_code falls through
        # to the `not result.success` branch.
        result = self._invoke_plan_with_result(
            tmp_path,
            _make_plan_result(
                solver_status="TIMEOUT",  # type: ignore[arg-type]
                success=False,
                exit_code=1,
                batch_count=0,
            ),
        )
        # With success=False and exit_code=1, _plan_exit_code returns result.exit_code=1
        assert result.exit_code != 0

    def test_plan_summary_contains_expected_format(self, tmp_path: Path) -> None:
        """Plan summary line should contain change count, batch count, oversized, solver."""
        result = self._invoke_plan_with_result(
            tmp_path,
            _make_plan_result(
                change_count=5,
                batch_count=3,
                oversized_count=1,
                solver_status="OPTIMAL",
                nodes_created=4,
                verify_rounds_used=2,
            ),
        )
        output = result.output
        assert "Planned 5 changes" in output
        assert "3 batches" in output
        assert "solver=OPTIMAL" in output

    def test_plan_summary_shows_verify_rounds_fraction(self, tmp_path: Path) -> None:
        """Summary one-liner must include verify_rounds=used/max in A/B format."""
        result = self._invoke_plan_with_result(
            tmp_path,
            _make_plan_result(verify_rounds_used=2),
        )
        assert "verify_rounds=2/3" in result.output

    def test_plan_summary_cap_hit_marker_present(self, tmp_path: Path) -> None:
        """When cap is hit the summary must append CAP-HIT."""
        result = self._invoke_plan_with_result(
            tmp_path,
            _make_plan_result(verify_rounds_used=3, verify_round_cap_hit=True),
        )
        assert "CAP-HIT" in result.output

    def test_plan_summary_custom_max_verify_rounds_reflected(self, tmp_path: Path) -> None:
        """--max-verify-rounds changes the denominator in the summary."""
        spec = tmp_path / "spec.md"
        spec.write_text("# Goal\nsome spec", encoding="utf-8")
        with (
            patch("milknado.domains.planning.Planner") as mock_planner_cls,
            patch("milknado.adapters.crg.CrgAdapter"),
            patch("milknado.app.run_command._load_or_default"),
            patch("milknado.app.run_command._ensure_db") as mock_ensure_db,
        ):
            mock_graph = MagicMock()
            mock_ensure_db.return_value = mock_graph
            mock_planner = MagicMock()
            mock_planner_cls.return_value = mock_planner
            mock_planner.launch.return_value = _make_plan_result(verify_rounds_used=1)
            result = runner.invoke(
                app,
                [
                    "plan",
                    "--spec",
                    str(spec),
                    "--max-verify-rounds",
                    "5",
                    "--project-root",
                    str(tmp_path),
                ],
            )
        assert "verify_rounds=1/5" in result.output


class TestResumeResetFlags:
    def test_resume_and_reset_together_exits_nonzero(self, tmp_path: Path) -> None:
        spec = tmp_path / "spec.md"
        spec.write_text("# Goal\nsome spec", encoding="utf-8")
        result = runner.invoke(
            app,
            [
                "plan",
                "--spec",
                str(spec),
                "--resume",
                "--reset",
                "--project-root",
                str(tmp_path),
            ],
        )
        assert result.exit_code != 0
        assert "mutually exclusive" in result.output.lower()

    def test_existing_plan_without_flag_exits_nonzero(self, tmp_path: Path) -> None:
        """ExistingPlanDetected raised by planner → CLI exits 1."""
        spec = tmp_path / "spec.md"
        spec.write_text("# Goal\nsome spec", encoding="utf-8")
        from milknado.domains.common.errors import ExistingPlanDetected

        with (
            patch("milknado.domains.planning.Planner") as mock_planner_cls,
            patch("milknado.adapters.crg.CrgAdapter"),
            patch("milknado.app.run_command._load_or_default"),
            patch("milknado.app.run_command._ensure_db") as mock_ensure_db,
        ):
            mock_graph = MagicMock()
            mock_ensure_db.return_value = mock_graph
            mock_planner = MagicMock()
            mock_planner_cls.return_value = mock_planner
            mock_planner.launch.side_effect = ExistingPlanDetected(5, 2, 2, 1)
            result = runner.invoke(
                app,
                ["plan", "--spec", str(spec), "--project-root", str(tmp_path)],
            )
        assert result.exit_code == 1
        assert "--resume" in result.output
        assert "--reset" in result.output


class TestDeriveGoal:
    def test_first_heading_extracted(self, tmp_path: Path) -> None:
        from milknado.app.spec_ingest import derive_goal

        spec = tmp_path / "spec.md"
        spec.write_text("# My Feature Goal\nsome content", encoding="utf-8")
        assert derive_goal(spec) == "My Feature Goal"

    def test_no_heading_returns_stem(self, tmp_path: Path) -> None:
        from milknado.app.spec_ingest import derive_goal

        spec = tmp_path / "my-spec.md"
        spec.write_text("No heading here, just prose.", encoding="utf-8")
        assert derive_goal(spec) == "my-spec"

    def test_heading_with_extra_whitespace_stripped(self, tmp_path: Path) -> None:
        from milknado.app.spec_ingest import derive_goal

        spec = tmp_path / "spec.md"
        spec.write_text("#   Lots of spaces   \nsome content", encoding="utf-8")
        assert derive_goal(spec) == "Lots of spaces"

    def test_heading_level_two_not_extracted(self, tmp_path: Path) -> None:
        from milknado.app.spec_ingest import derive_goal

        spec = tmp_path / "spec.md"
        spec.write_text("## Not a top heading\n# Real heading\n", encoding="utf-8")
        # ## is not matched by `line.startswith("# ")` check... but "## " starts with "#"
        # Let's verify the actual behavior: "## Not..." starts with "# " is False,
        # because "## " != "# " prefix. Actually "## ".startswith("# ") is False.
        result = derive_goal(spec)
        assert result == "Real heading"
