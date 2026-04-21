from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from milknado.domains.graph import MikadoGraph
from milknado.domains.planning.planner import Planner


def _make_manifest(changes: list[dict]) -> str:  # type: ignore[type-arg]
    payload = {
        "manifest_version": "milknado.plan.v2",
        "goal": "test goal",
        "goal_summary": "test summary",
        "changes": changes,
    }
    return "```json\n" + json.dumps(payload) + "\n```"


@pytest.fixture()
def tmp_graph(tmp_path: Path) -> MikadoGraph:
    return MikadoGraph(tmp_path / "test.db")


@pytest.fixture()
def mock_crg() -> MagicMock:
    crg = MagicMock()
    crg.semantic_search_nodes.return_value = []
    return crg


class TestUS009CoverageLoop:
    """US-009: coverage-check loop integration — PlanResult telemetry accuracy."""

    def _covered(self, us: str) -> str:
        return _make_manifest(
            [
                {"id": "c1", "path": "src/foo.py", "description": f"{us} implement"},
                {"id": "c2", "path": "tests/test_foo.py", "description": f"{us} tests"},
            ]
        )

    def _uncovered(self, us: str) -> str:
        return _make_manifest(
            [
                {"id": "c1", "path": "src/foo.py", "description": f"{us} implement"},
            ]
        )

    def _degrade(self) -> MagicMock:
        from milknado.domains.common.types import DegradationMarker

        m = MagicMock()
        m.structural_map.return_value = DegradationMarker(source="tilth", reason="mocked")
        return m

    @patch("milknado.domains.planning.planner.TilthAdapter")
    @patch("milknado.domains.planning.planner.subprocess.run")
    def test_full_coverage_on_first_pass_no_reprompt(
        self,
        mock_run: MagicMock,
        mock_tilth_cls: MagicMock,
        tmp_path: Path,
        tmp_graph: MikadoGraph,
        mock_crg: MagicMock,
    ) -> None:
        mock_tilth_cls.return_value = self._degrade()
        spec = tmp_path / "spec.md"
        spec.write_text("US-001 should add feature X", encoding="utf-8")
        mock_run.return_value = MagicMock(returncode=0, stdout=self._covered("US-001"))

        result = Planner(tmp_graph, mock_crg, "claude").launch(
            "goal", tmp_path, spec_path=spec, max_verify_rounds=3
        )

        assert mock_run.call_count == 1
        assert result.verify_rounds_used == 1
        assert result.coverage_gaps_remaining == 0
        assert result.verify_round_cap_hit is False

    @patch("milknado.domains.planning.planner.TilthAdapter")
    @patch("milknado.domains.planning.planner.subprocess.run")
    def test_one_us_missing_tests_reprompts_and_succeeds(
        self,
        mock_run: MagicMock,
        mock_tilth_cls: MagicMock,
        tmp_path: Path,
        tmp_graph: MikadoGraph,
        mock_crg: MagicMock,
    ) -> None:
        mock_tilth_cls.return_value = self._degrade()
        spec = tmp_path / "spec.md"
        spec.write_text("US-001 should add feature X", encoding="utf-8")
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout=self._uncovered("US-001")),
            MagicMock(returncode=0, stdout=self._covered("US-001")),
        ]

        result = Planner(tmp_graph, mock_crg, "claude").launch(
            "goal", tmp_path, spec_path=spec, max_verify_rounds=3
        )

        assert mock_run.call_count == 2
        assert result.verify_rounds_used == 2
        assert result.coverage_gaps_remaining == 0
        assert result.verify_round_cap_hit is False

    @patch("milknado.domains.planning.planner.TilthAdapter")
    @patch("milknado.domains.planning.planner.subprocess.run")
    def test_coverage_cap_hit_raises_insufficient_coverage(
        self,
        mock_run: MagicMock,
        mock_tilth_cls: MagicMock,
        tmp_path: Path,
        tmp_graph: MikadoGraph,
        mock_crg: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        from milknado.domains.common.errors import InsufficientTestCoverageError

        mock_tilth_cls.return_value = self._degrade()
        spec = tmp_path / "spec.md"
        spec.write_text("US-001 should add feature X", encoding="utf-8")
        mock_run.return_value = MagicMock(returncode=0, stdout=self._uncovered("US-001"))

        with caplog.at_level(logging.WARNING, logger="milknado"):
            with pytest.raises(InsufficientTestCoverageError) as exc_info:
                Planner(tmp_graph, mock_crg, "claude").launch(
                    "goal", tmp_path, spec_path=spec, max_verify_rounds=1
                )

        assert "src/foo.py" in exc_info.value.orphan_changes
        assert any("verify-spec round cap hit" in r.message for r in caplog.records)

    @patch("milknado.domains.planning.planner.TilthAdapter")
    @patch("milknado.domains.planning.planner.subprocess.run")
    def test_impl_without_us_ref_is_orphan_when_no_test_coverage(
        self,
        mock_run: MagicMock,
        mock_tilth_cls: MagicMock,
        tmp_path: Path,
        tmp_graph: MikadoGraph,
        mock_crg: MagicMock,
    ) -> None:
        from milknado.domains.common.errors import InsufficientTestCoverageError

        mock_tilth_cls.return_value = self._degrade()
        spec = tmp_path / "spec.md"
        spec.write_text("This spec has no user story refs at all.", encoding="utf-8")
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=_make_manifest([{"id": "c1", "path": "src/foo.py", "description": "add foo"}]),
        )

        with pytest.raises(InsufficientTestCoverageError) as exc_info:
            Planner(tmp_graph, mock_crg, "claude").launch(
                "goal", tmp_path, spec_path=spec, max_verify_rounds=3
            )

        assert "src/foo.py" in exc_info.value.orphan_changes
