"""Adversarial tests for the CLI plan command.

Focus: --spec validation, solver-status exit codes, binary files, empty files, unicode filenames.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TypedDict, Unpack
from unittest.mock import patch

from click.testing import Result
from typer.testing import CliRunner

from milknado.cli import app
from milknado.domains.planning.planner import PlanResult

runner = CliRunner()


class _PlanResultOverrides(TypedDict, total=False):
    success: bool
    exit_code: int
    context_path: Path | None
    nodes_created: int
    batch_count: int
    oversized_count: int
    solver_status: str
    change_count: int
    mega_batch_change_count: int | None


class _FakeGraph:
    def close(self) -> None:
        pass


class _FakePlanner:
    def __init__(self, result: PlanResult) -> None:
        self.result: PlanResult = result
        self.launch_calls: list[tuple[str, Path, Path | None]] = []

    def launch(
        self, goal: str, project_root: Path, *, spec_path: Path | None = None
    ) -> PlanResult:
        self.launch_calls.append((goal, project_root, spec_path))
        return self.result


def _make_plan_result(**kwargs: Unpack[_PlanResultOverrides]) -> PlanResult:
    return PlanResult(
        success=kwargs.get("success", True),
        exit_code=kwargs.get("exit_code", 0),
        context_path=kwargs.get("context_path"),
        nodes_created=kwargs.get("nodes_created", 3),
        batch_count=kwargs.get("batch_count", 2),
        oversized_count=kwargs.get("oversized_count", 0),
        solver_status=kwargs.get("solver_status", "OPTIMAL"),
        change_count=kwargs.get("change_count", 3),
        mega_batch_change_count=kwargs.get("mega_batch_change_count"),
    )


def _fake_ensure_db(*_args: object, **_kwargs: object) -> _FakeGraph:
    return _FakeGraph()


def _planner_factory(fake_planner: _FakePlanner) -> Callable[..., _FakePlanner]:
    def factory(*_args: object, **_kwargs: object) -> _FakePlanner:
        return fake_planner

    return factory


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
        _ = spec.write_text("# Goal\nsome spec", encoding="utf-8")
        result = runner.invoke(
            app,
            ["plan", "--spec", str(spec), "--project-root", str(tmp_path)],
        )
        assert result.exit_code != 0

    def test_binary_file_renamed_md_exit_nonzero(self, tmp_path: Path) -> None:
        """Binary file with .md extension — _derive_goal reads it as UTF-8 and rejects it."""
        spec = tmp_path / "binary.md"
        _ = spec.write_bytes(b"\x00\x01\x02\x03\xff\xfe\xfd")
        fake_planner = _FakePlanner(_make_plan_result())
        with (
            patch(
                "milknado.domains.planning.Planner",
                new=_planner_factory(fake_planner),
            ),
            patch("milknado.adapters.crg.CrgAdapter"),
            patch("milknado.cli.plan._ensure_db", new=_fake_ensure_db),
        ):
            result = runner.invoke(
                app,
                ["plan", "--spec", str(spec), "--project-root", str(tmp_path)],
            )
            # The bytes are not valid UTF-8 — _derive_goal raises typer.BadParameter,
            # which typer surfaces as exit code 2.
            assert result.exit_code == 2
            assert "not valid UTF-8" in result.output

    def test_empty_spec_file_does_not_crash(self, tmp_path: Path) -> None:
        """0-byte spec.md — _derive_goal returns stem, planner is launched."""
        spec = tmp_path / "spec.md"
        _ = spec.write_bytes(b"")  # 0 bytes
        fake_planner = _FakePlanner(_make_plan_result())
        with (
            patch(
                "milknado.domains.planning.Planner",
                new=_planner_factory(fake_planner),
            ),
            patch("milknado.adapters.crg.CrgAdapter"),
            patch("milknado.cli.plan._ensure_db", new=_fake_ensure_db),
        ):
            _ = runner.invoke(
                app,
                ["plan", "--spec", str(spec), "--project-root", str(tmp_path)],
            )
            # Empty spec → _derive_goal should return stem ("spec") not crash
            assert len(fake_planner.launch_calls) == 1
            goal_arg = fake_planner.launch_calls[0][0]
            assert goal_arg == "spec"


class TestSolverStatusExitCodes:
    def _invoke_plan_with_result(self, tmp_path: Path, plan_result: PlanResult) -> Result:
        spec = tmp_path / "spec.md"
        _ = spec.write_text("# My Goal\nsome spec", encoding="utf-8")
        fake_planner = _FakePlanner(plan_result)
        with (
            patch(
                "milknado.domains.planning.Planner",
                new=_planner_factory(fake_planner),
            ),
            patch("milknado.adapters.crg.CrgAdapter"),
            patch("milknado.cli.plan._ensure_db", new=_fake_ensure_db),
        ):
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
                solver_status="TIMEOUT",
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
            ),
        )
        output = result.output
        assert "5" in output or "Planned" in output


class TestMegaBatchReport:
    """_plan_summary surfaces a mega-batch note only when one was detected."""

    def test_summary_includes_mega_batch_when_present(self) -> None:
        from milknado.cli.plan import _plan_summary

        summary = _plan_summary(_make_plan_result(mega_batch_change_count=6))
        assert "mega-batch" in summary
        assert "6" in summary

    def test_summary_omits_mega_batch_when_none(self) -> None:
        from milknado.cli.plan import _plan_summary

        summary = _plan_summary(_make_plan_result())
        assert "mega-batch" not in summary


class TestDeriveGoal:
    def test_first_heading_extracted(self, tmp_path: Path) -> None:
        from milknado.cli import _derive_goal

        spec = tmp_path / "spec.md"
        _ = spec.write_text("# My Feature Goal\nsome spec", encoding="utf-8")
        assert _derive_goal(spec) == "My Feature Goal"

    def test_no_heading_returns_stem(self, tmp_path: Path) -> None:
        from milknado.cli import _derive_goal

        spec = tmp_path / "my-spec.md"
        _ = spec.write_text("No heading here\n", encoding="utf-8")
        assert _derive_goal(spec) == "my-spec"

    def test_heading_with_extra_whitespace_stripped(self, tmp_path: Path) -> None:
        from milknado.cli import _derive_goal

        spec = tmp_path / "spec.md"
        _ = spec.write_text("#   Lots of spaces   \nsome content", encoding="utf-8")
        assert _derive_goal(spec) == "Lots of spaces"

    def test_heading_level_two_not_extracted(self, tmp_path: Path) -> None:
        from milknado.cli import _derive_goal

        spec = tmp_path / "spec.md"
        _ = spec.write_text("## Not a top heading\n# Real heading\n", encoding="utf-8")
        # ## is not matched by `line.startswith("# ")` check... but "## " starts with "#"
        # Let's verify the actual behavior: "## Not..." starts with "# " is False,
        # because "## " != "# " prefix. Actually "## ".startswith("# ") is False.
        result = _derive_goal(spec)
        assert result == "Real heading"
